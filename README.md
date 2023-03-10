# ChebyLighter: Optimal Curve Estimation for Low-light Image Enhancement, ACM MM 2022

[[Paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548135)

## Codes

### Requirements
* python==3.7
* pytorch==1.7.1
* torchvision==0.8.2
* torchsummaryx==1.3.0
* kornia==0.5.0
* pillow==9.0.1
### Dataset used in the paper

### Training 
```bash
python runLOL.py --gpu_id 1 --num_orders 6 --config configs/LOL;
```

### Testing
```bash
python test_one_image.py --img_path TSET_IMG_PATH/test_image.png
```

## Citation
```bibtex
@inproceedings{chebylighter_2022_ACMMM,
author = {Pan, Jinwang and Zhai, Deming and Bai, Yuanchao and Jiang, Junjun and Zhao, Debin and Liu, Xianming},
title = {ChebyLighter: Optimal Curve Estimation for Low-Light Image Enhancement},
year = {2022},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3503161.3548135},
pages = {1358–1366},
numpages = {9},
series = {MM '22}
}
```