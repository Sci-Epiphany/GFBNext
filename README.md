

## Global feature-base multimodal semantic segmentation

## Introduction
This is an introduction to the initial version of GFBN. GFBN is used for dual-branch multi-modal semantic segmentation. The author's main innovation lies in proposing the Cross-attention correction module and the feature fusion module of large convolution kernel. The code is mainly based on CMNext and CMX. If you want to use this code, please quote the two articles in the last page.

## Updates
- [x] 010/2023, init repository.
- [x] 



## Data preparation
Prepare three datasets:
- [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), for RGB-Depth semantic segmentation.
- [MFNet](https://github.com/haqishen/MFNet-pytorch), for RGB-Thermal semantic segmentation.
- [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation), for multimodal material segmentation with RGB-A-D-N modalities.

Then, all datasets are structured as:

```
data/
├── NYUDepthv2
│   ├── RGB
│   ├── HHA
│   └── Label
├── MFNet
│   ├── rgb
│   ├── ther
│   └── labels
├── MCubeS
│   ├── polL_color
│   ├── polL_aolp
│   ├── polL_dolp
│   ├── NIR_warped
│   └── SS
```

## Model Zoo

### MCubeS
GFBN models checkpoints is available at ([**GoogleDrive**](https://drive.google.com/drive/folders/1fJwS-EGc_LKxC28zUTp9afeQVEgnzIh3?usp=sharing)).




## Training

Before training, please download [pre-trained SegFormer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing), such as `checkpoints/pretrained/segformer/mit_b2.pth`.

```text
checkpoints/pretrained/segformer
├── mit_b2.pth
└── mit_b4.pth
```

To train GFBN model, please use change yaml file for `--cfg`. Several training examples using 4 A100 GPUs are:  

```bash
cd path/to/GFBN
export PYTHONPATH="path/to/GFBN"
python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/train_mm.py --cfg configs/nyu_rgbd.yaml
python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/train_mm.py --cfg configs/mcubes_rgbadn_next.yaml
```


## Evaluation
To evaluate GFBN models on the MCubeS, please download respective model weights ([**GoogleDrive**](https://drive.google.com/drive/folders/1fJwS-EGc_LKxC28zUTp9afeQVEgnzIh3?usp=sharing)) as:


```text
output/
├── MCubeSBGM_GFBNEXT
│   ├── GFBNEXT_B2_MCubeSBGM_in.pth
│   ├── GFBNEXT_B2_MCubeSBGM_id.pth
│   ├── GFBNEXT_B2_MCubeSBGM_iadn.pth


Then, modify `--cfg` to respective config file, and run:
```bash
cd path/to/GFBNext
export PYTHONPATH="path/to/GFBNext"
CUDA_VISIBLE_DEVICES=0 python tools/val_mm.py --cfg configs/mcubes_rgbadn_next.yaml
```

## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

If you use GFBN model, please cite the following works:

- **DeLiVER & CMNeXt** [[**PDF**](https://arxiv.org/pdf/2303.01480.pdf)]
```
@article{zhang2023delivering,
  title={Delivering Arbitrary-Modal Semantic Segmentation},
  author={Zhang, Jiaming and Liu, Ruiping and Shi, Hao and Yang, Kailun and Reiß, Simon and Peng, Kunyu and Fu, Haodong and Wang, Kaiwei and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2303.01480},
  year={2023}
}
```

- **CMX** [[**PDF**](https://arxiv.org/pdf/2203.04838.pdf)]
```
@article{liu2022cmx,
  title={CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers},
  author={Liu, Huayao and Zhang, Jiaming and Yang, Kailun and Hu, Xinxin and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2203.04838},
  year={2022}
}
```
