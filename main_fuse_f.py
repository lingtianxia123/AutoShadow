import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from src.IFNet_f import build_model
from src.HAUNet import HAUNet
from src.IFNet_p import IFNet_Param
from src.dataset import build_dataset
from mindspore.communication import init, get_rank, get_group_size

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    parser = argparse.ArgumentParser('Set params', add_help=False)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.0, type=float, help='gradient clipping max norm')

    # model
    parser.add_argument('--model', default='IFNet_f', type=str, help="Name of the model to use")

    # loss weights
    parser.add_argument('--avg_weight', default=1.0, type=float)
    parser.add_argument('--min_weight', default=1.0, type=float)
    parser.add_argument('--fuse_weight', default=1.0, type=float)

    # dataset parameters
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--weights_mask', default='', help='load weights')
    parser.add_argument('--weights_param', default='', help='load weights')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default=False, type=bool)
    parser.add_argument('--display_freq', default=100, type=int)

    return parser


class WithLossCell(nn.Cell):
    def __init__(self, network, loss_fun):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fun = loss_fun

    def construct(self, deshadow_img, fg_instance, pred_mask, pred_param, shadow_img):
        fuse_img = self.network(deshadow_img, fg_instance, pred_mask, pred_param)
        loss = self.loss_fun(fuse_img, shadow_img)
        return loss


class WithEvalCell(nn.Cell):
    def __init__(self, network, loss_fun):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fun = loss_fun

    def construct(self, deshadow_img, fg_instance, pred_mask, pred_param, shadow_img):
        fuse_img = self.network(deshadow_img, fg_instance, pred_mask, pred_param)
        loss = self.loss_fun(fuse_img, shadow_img)
        return fuse_img, loss


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, args):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.args = args

    def construct(self, deshadow_img, fg_instance, pred_mask, pred_param, shadow_img):
        loss = self.network(deshadow_img, fg_instance, pred_mask, pred_param, shadow_img)
        grads = self.grad(self.network, self.weights)(deshadow_img, fg_instance, pred_mask, pred_param, shadow_img)
        if self.args.clip_max_norm > 0:
            grads = ops.clip_by_global_norm(grads, self.args.clip_max_norm)
        self.optimizer(grads)
        return loss


def main(args):
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    init("nccl")
    value = get_rank()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.HYBRID_PARALLEL)

    np.random.seed(args.seed)
    ms.common.set_seed(1)

    # dataset
    data_loader_train = build_dataset(image_set='train', args=args)
    data_loader_bos = build_dataset(image_set='bos', args=args)
    data_loader_bosfree = build_dataset(image_set='bosfree', args=args)

    # network   mask
    network_mask = HAUNet()
    if args.weights_mask:
        param_dict = ms.load_checkpoint(args.weights_mask)
        ms.load_param_into_net(network_mask, param_dict)
        print("load mask weight:", args.weights_mask)
    network_mask.set_grad(False)

    # network   param
    network_param = IFNet_Param()
    if args.weights_param:
        param_dict = ms.load_checkpoint(args.weights_param)
        ms.load_param_into_net(network_param, param_dict)
        print("load param weight:", args.weights_param)
    network_param.set_grad(False)

    # network  fuse
    network, loss_fun, evaluator = build_model(args)
    network_loss = WithLossCell(network, loss_fun)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay, beta1=0.5, beta2=0.999)

    # resume
    if args.resume:
        checkpoint_path = os.path.join(args.resume, "checkpoint.ckpt")
        if os.path.exists(checkpoint_path):
            ms.load_param_into_net(network, ms.load_checkpoint(checkpoint_path))
            print("resume network:", checkpoint_path)
        else:
            print("no network checkpoint:", checkpoint_path)
        optimizer_path = os.path.join(args.resume, "optimizer.ckpt")
        if os.path.exists(optimizer_path):
            ms.load_param_into_net(optimizer, ms.load_checkpoint(optimizer_path))
            print("resume optimizer:", optimizer_path)
        else:
            print("no optimizer checkpoint:", optimizer_path)

    trainOneStep = TrainOneStepCell(network_loss, optimizer, args)
    evalOneStep = WithEvalCell(network, loss_fun)

    steps = data_loader_train.get_dataset_size()
    for epoch in range(args.epochs):
        step = 0
        loss_train = 0
        trainOneStep.set_train()
        for data in data_loader_train.create_dict_iterator():
            shadow_img = data['shadow_img']
            deshadow_img = data['deshadow_img']
            fg_instance = data['fg_instance']
            pred_mask = network_mask(deshadow_img, fg_instance)
            pred_param = network_param(deshadow_img, fg_instance, pred_mask)
            loss = trainOneStep(deshadow_img, fg_instance, pred_mask, pred_param, shadow_img)
            if step % args.display_freq == 0:
                print(f"Epoch: [{epoch} / {args.epochs}], "f"step: [{step} / {steps}], " f"loss: {loss}")
            step = step + 1
            loss_train += loss.asnumpy()

        evalOneStep.set_grad(False)
        evaluator.clear()
        bos_loss = 0
        for data in data_loader_bos.create_dict_iterator():
            shadow_img = data['shadow_img']
            deshadow_img = data['deshadow_img']
            fg_instance = data['fg_instance']
            fg_shadow = data['fg_shadow']
            pred_mask = network_mask(deshadow_img, fg_instance)
            pred_param = network_param(deshadow_img, fg_instance, pred_mask)
            fuse_img, loss = evalOneStep(deshadow_img, fg_instance, pred_mask, pred_param, shadow_img)
            evaluator.update(pred_mask, fuse_img, fg_shadow, shadow_img)
            bos_loss += loss.asnumpy()
        bos_out = evaluator.eval()
        bos_num = evaluator._samples_num

        evaluator.clear()
        bosfree_loss = 0
        for data in data_loader_bosfree.create_dict_iterator():
            shadow_img = data['shadow_img']
            deshadow_img = data['deshadow_img']
            fg_instance = data['fg_instance']
            fg_shadow = data['fg_shadow']
            pred_mask = network_mask(deshadow_img, fg_instance)
            pred_param = network_param(deshadow_img, fg_instance, pred_mask)
            fuse_img, loss = evalOneStep(deshadow_img, fg_instance, pred_mask, pred_param, shadow_img)
            evaluator.update(pred_mask, fuse_img, fg_shadow, shadow_img)
            bosfree_loss += loss.asnumpy()
        bosfree_out = evaluator.eval()
        bosfree_num = evaluator._samples_num

        log_msg = f"Epoch: [{epoch}/{args.epochs}], "f"loss_train: {loss_train}, "f"bos_loss: [{bos_loss} / {bos_num}]"
        for k, v in bos_out.items():
            log_msg += ', ' + k + ":%0.5f" % v
        log_msg += f", bosfree_loss: [{bosfree_loss} / {bosfree_num}]"
        for k, v in bosfree_out.items():
            log_msg += ', ' + k + ":%0.5f" % v
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(log_msg + "\n")
        print(log_msg)
        
        ms.save_checkpoint(network, os.path.join(args.output_dir, "checkpoint.ckpt"))
        ms.save_checkpoint(optimizer, os.path.join(args.output_dir, "optimizer.ckpt"))
        if args.output_dir:
            if (epoch + 1) % 50 == 0 or epoch > args.epochs - 1:
                ms.save_checkpoint(network, os.path.join(args.output_dir, "checkpoint_%05d.ckpt" % epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        if args.resume:
            args.output_dir = args.resume[:-len(args.resume.split('/')[-1]) - 1]
        else:
            args.output_dir = args.output_dir + '/' + str(args.model) + '/' + 'lr_' + str(args.lr) + '_bs_' + str(args.batch_size) + '_epochs_' + str(args.epochs)
            args.output_dir = args.output_dir + '_' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
            print("output_dir:", args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
    main(args)
