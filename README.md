# SAM-PM: Enhancing Video Camouflaged Object Detection using Spatio-Temporal Attention


## Installation

The current code was written and tested on top of [```pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel```]([https://hub.docker.com/layers/pytorch/pytorch/2.1.2-cuda12.1-cudnn8-runtime/images/sha256:3387e598cb94fc248d82e712a65b10931a990cea3a2e76362ca30d135f565de4](https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-devel/images/sha256-4f66166dd757752a6a6a9284686b4078e92337cd9d12d2e14d2d46274dfa9048)) docker. To install the dependencies, run the following:
```sh
pip install -r requirements.txt
```

## Dataset
Download and setup the dataset using following cmd:
```sh
bash dataset_download.sh
```


## Train

To train the model run both Stage 1 and 2 sequentially:
### Stage 1:
```sh
python train.py
```
### Stage 2:
Modify these parameters in ```config.py```

```js
num_epochs: 140
save_log_weights_interval: 20
train_metric_interval: 20
learning_rate: 5e-4
steps: []
stage1: True
```

and run:
```sh
python train.py
```

## Test

To test the model run

```sh
python test.py --ckpt {Path to checkpoint}
```
Benchmark scores mentioned in the paper uses SLT's evaluation code which is given at ```eval/```


For CAD/frog alone delete groundtruth images from 021_gt.png onwards since those are empty masks and they throw an error with the matlab code



Change the following paths according to where you are saving the predictions and where you have placed the dataset for ```main_CAD.m``` and ```main_MoCA.m```:
```Matlab
resPath = ['../best/' seqfolder '/'] % Enter the path of the results
```
Before running the Matlab scripts make sure you have [Deep Learning Tool box](https://www.mathworks.com/products/deep-learning.html) installed in Matlab. 
Run the following Matlab scripts:
 ```
main_CAD.m
main_MoCA.m
```
