<<<<<<< HEAD
# tire_recognition_CAMN

# tire_recognition_CAMN
# 环境要求
matplotlib==3.7.4
matplotlib-inline==0.1.6
mdurl==0.1.2
mmcls==0.23.0
mmcv-full==1.5.2
mmdet==2.24.1
numpy==1.24.4
# 创建环境
conda create -n tire python=3.8.2 -y
conda activate tire
# 安装基础环境
pip install -r requirement.txt #安装基础环境
# 搭建环境
cd tire_recognition_CAMN
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

# 训练过程(5way1shot训练为例)
python tools/classification/train.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py 

# 测试过程(5way1shot训练为例)
python tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py --metric Recall

参考代码库：
https://github.com/open-mmlab/mmfewshot

# tirerecognition_DAE-MN
>>>>>>> 8e1884db7667f9e999fac591ec1b8dcb11aec17d
