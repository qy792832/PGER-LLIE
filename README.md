#  Retinex-Consistent and Time-Aware Diffusion for Low-Light Image Enhancement
### [Paper]() | [Code]([https://github.com/XunpengYi/Diff-Retinex-Plus](https://github.com/qy792832/PGER-LLIE)) 

**Retinex-Consistent and Time-Aware Diffusion for Low-Light Image Enhancement**
Yi Qu, Senming Zhong, Minglong Xue


![Framework](asserts/framework.png)

## 1. Create Environment
- Create Conda Environment
```
conda env create -f environment.yaml
conda activate diff_retinex_plus_env
```
- Install BasicSR
```
pip install basicsr==1.3.4.4
```

## 2. Prepare Your Dataset
Please download the datasets from the following links: [lolv1](https://daooshee.github.io/BMVC2018website/), [lolv2 real/syn](https://github.com/flyywh/CVPR-2020-Semi-Low-Light), [LSRW](https://github.com/JianghaiSCU/R2RNet#dataset).

For the LLRW dataset, you can download it from [[Google Drive]](https://drive.google.com/drive/folders/19Ulq-hanwKitZvp3LpXTuNa8mxNWVJym?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1ylwJDFp-CHZ6NizHwtsZuw) (code: ra4u).

The A-LLIE dataset is composed of challenging scenes collected from existing datasets and requires a redistribution license. You can obtain it from the corresponding original sources.

## 3. Pretrained Weights
Please download the weights and place them in the `pretrained_weights` folder.

The pretrained weight for lolv1 dataset is at [[Google Drive]](https://drive.google.com/drive/folders/1noF5hwnduC2-IG5FaSYZr303u3atWYM-?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1r_7vjpS6XO7saemlQlYepQ) (code: 1ucp).

The pretrained weight for lolv2 real dataset is at [[Google Drive]](https://drive.google.com/drive/folders/1HVcX2YW5XKjKIiqKibJMmYV1bCorswxt?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1maR_XS0rpbgDkyOi2UYbsw) (code: vmwa).

The pretrained weight for lolv2 syn dataset is at [[Google Drive]](https://drive.google.com/drive/folders/1nvdPQOIRcuidYwkXBsG2KDs7eLH-ErmL?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1gLidotuCYLFx7bIXxTNR6w) (code: kzii).

The pretrained weight for LSRW dataset is at [[Google Drive]](https://drive.google.com/drive/folders/19q3a1cSu51Id9MHfn8xPVsS4Aimv5gQw?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1bIv1NRdEQEqnDF7GevCZsQ) (code: 21gt).

The pretrained weight for A-LLIE dataset is at [[Google Drive]](https://drive.google.com/drive/folders/1qUfOM_1c4UEbrZv1F9NTwD_nCDcUiD88?usp=sharing) | [[Baidu Drive]](https://pan.baidu.com/s/1p04RgJ6utQ58irSzySInyg) (code: ymjd).

