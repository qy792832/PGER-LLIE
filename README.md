#  Retinex-Consistent and Time-Aware Diffusion for Low-Light Image Enhancement
### [Paper]() | [Code]([https://github.com/XunpengYi/Diff-Retinex-Plus](https://github.com/qy792832/PGER-LLIE)) 

**Retinex-Consistent and Time-Aware Diffusion for Low-Light Image Enhancement**
Yi Qu, Senming Zhong, Minglong Xue


![Framework](image/kuangjian.png)

## How to Run the Code?
* conda activate PGER
* pip install -r requirements
### Dependencies

* OS: Ubuntu 22.04
* nvidia:
	- cuda: 12.1
* python 3.9

### Data Preparation

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
- [LRSW](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)(code: wmrr)

## Train
```python train.py ```

## Test
```python evaluate.py ```



