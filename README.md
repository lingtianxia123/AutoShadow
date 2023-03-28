# Automatic Shadow Generation via Exposure Fusion

Official MindSpore code for paper 'Automatic Shadow Generation via Exposure Fusion'

We propose an automatic shadow generation method, which consists of Hierarchy Attention U-Net (HAU-Net) and Illumination-Aware Fusion Network (IFNet).

[[Paper]](https://ieeexplore.ieee.org/document/10043015)

## Pipeline
![image](https://github.com/lingtianxia123/AutoShadow/blob/main/images/framework.png)

### Install

- Clone repo

  ```
  git clone https://github.com/lingtianxia123/AutoShadow
  cd AutoShadow
  ```
- Install dependencies ( [MindSpore](https://www.mindspore.cn/en), scikit-learn, opencv-python, pandas. Recommend to use [Anaconda](https://www.anaconda.com/).)

  ```
  # Create a new conda environment
  conda create -n shadow python=3.7
  conda activate shadow
  
  # Install [MindSpore](https://www.mindspore.cn/en)
    
  # Install other packages
  pip install -r requirements.txt
  ```

## Environment

## Dataset


### Citation

  If you found this code useful please cite our work as:

  ```
  @ARTICLE{Meng_2023_TMM,
    author={Meng, Quanling and Zhang, Shengping and Li, Zonglin and Wang, Chenyang and Zhang, Weigang and Huang, Qingming},
    journal={IEEE Transactions on Multimedia}, 
    title={Automatic Shadow Generation via Exposure Fusion}, 
    year={2023},
    volume={},
    number={},
    pages={1-13},
    doi={10.1109/TMM.2023.3244398}
    }
    ```
