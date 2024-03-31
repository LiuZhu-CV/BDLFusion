# BDLFusion
 Codes of Bi-level Dynamic Learning  for Jointly Multi-modality Image Fusion and Beyond


Zhu Liu,
Jinyuan Liu,
Guanyao Wu,
Long Ma,
Xin Fan,
Risheng Liu. 
In IJCAI 2023. 
[Paper](https://arxiv.org/pdf/2305.06720.pdf)

## Fusion Reuslts and Chinese Version

The source images and fused results on TNO, RoadScene and M3FD (600 images) are
provided in [link](https://drive.google.com/drive/folders/1UI5KTL5FFDtbmSMEmB-4CCe3xrdMc0_3)

中文版介绍提供在此链接 [link](https://arxiv.org/pdf/2308.03979.pdf)

Welcome all comparision and disscussion!
If you have any questions, please sending an email to "liuzhu_ssdut@foxmail.com"


## Requirements
+ Python 3.7
+ PyTorch 1.10.1
+ Checkpoint of detection 
[Checkpoint](https://drive.google.com/file/d/1jZgtcS9dxm4fCDutFk_zPqaeZuDlJa5C/view?usp=share_link)

## Usage

### Testing
Run "**python test.py**" to test the model.


### Training

Run "**python train.py**" to train the model.

The warmstart checkpoint and saliency weight maps are provided at url (https://drive.google.com/drive/folders/1WxxuXFDX4-18DfAJjDWoemREBDRcEaDH?usp=drive_link)

### Workflow

<img src='imgs/first.png' width=840> 

### Results of detection
<img src='imgs/detec.png' width=840> 

### Results of segmentation
<img src='imgs/Seg.png' width=840> 

## Citation
If you use this code for your research, please cite our paper.

```
@article{liu2023bilevel,
  title={Bi-level Dynamic Learning  for Jointly Multi-modality Image Fusion and Beyond},
  author={Zhu Liu and Jinyuan Liu and Guanyao Wu and Long Ma and Xin Fan and Risheng Liu},
  journal={IJCAI},
  year={2023},
}
```













