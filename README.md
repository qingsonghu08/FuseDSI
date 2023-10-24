![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# Diverse Semantic Information Fusion for Unsupervised Person Re-Identification [[pdf]](wating)
The *official* repository for [Diverse Semantic Information Fusion for Unsupervised Person Re-Identification](wating).

## Requirements

### Installation
```bash
pip install -r requirements.txt
```
We recommend to use /Python=3.8 /torch=1.10.1 /torchvision=0.11.2 /timm=0.6.13 /cuda==11.3 /faiss-gpu=1.7.2/ 24G RTX 3090 or RTX 4090 for training and evaluation. If you find some packages are missing, please install them manually. 


### Prepare Datasets

```bash
mkdir data
```

Download the datasets:
- [Market-1501](https://drive.google.com/file/d/1pYM3wruB8TonHLwMQ_g1KAz-UqRrH006/view?usp=drive_link)
- [MSMT17](https://drive.google.com/file/d/1TD3COX3laYIpXNvKN6vazv_7x8PNdYkI/view?usp=drive_link)
- [LUPerson](https://github.com/DengpanFu/LUPerson).  
- We don't have the copyright of the LUPerson dataset. Please contact authors of LUPerson to get this dataset.
- You can download the file list ordered by the CFS score for the LUPerson. [[CFS_list.pkl]](https://drive.google.com/file/d/1D6RaiOv3F2WSABYfQB1Aa88mwGoVNa3k/view?usp=sharing)

Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── MSMT17
│   └── train
│   └── test
│   └── ..
└── VeRi
    └── image_train
    └── image_query
    └── image_test
    └── *.txt
```

## Pre-trained Models 
|      Model      | Download |
|:---------------:| :------: |
|    ViT-S/16     | [link](https://drive.google.com/file/d/15iOMEE6CL5Z__WyOLgNGwxPiewwwsaAT/view?usp=drive_link) |
|  ViT-S/16+ICS   | [link](https://drive.google.com/file/d/18FL9JaJNlo15-UksalcJRXX-0dgo4Mz4/view?usp=sharing) |
|  ViT-B/16+ICS   | [link](https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view?usp=sharing) |

## Baseline Pre-trained 30 epochs Models 
|   Model    | Download |
|:----------:| :------: |
| Market1501 | [link](https://drive.google.com/file/d/1rni24KS4B7YvpEnrNFyVP7UXUOie1hrA/view?usp=drive_link) |
|   MSMT17   | [link](https://drive.google.com/file/d/1MppUc-i9LbeCXLqQyTvw_L_i88mNEyZr/view?usp=drive_link) |

Please download pre-trained models and put them into your custom file path.

## Examples
You can use 1 or 2 GPUs for training. For more parameter configuration, please check **`market_usl.sh`**, **`msmt_usl.sh`**.

`sh market_usl.sh`

`sh msmt_usl.sh`


## ReID performance

We have reproduced the performance to verify the reproducibility. The reproduced results may have a gap of about 1.0% with the numbers in the paper.

### USL ReID
  
##### Market-1501
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:----------:|:------: |
| ViT-S/16      | 256*128 | 88.5/94.6  |[model](https://drive.google.com/file/d/1lD8zV6ROGWM9syfK_sarIK13ICTPwzhp/view?usp=drive_link)|


##### MSMT17
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:----------:|:------: |
| ViT-S/16      | 256*128 | 52.2/76.9  |[model](https://drive.google.com/file/d/1e_YT7iqBV9OhF-waRtJapt78s9Sfivaj/view?usp=drive_link)|


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[LUPerson](https://github.com/DengpanFu/LUPerson), [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid),
[TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)

## Citation

If you find this code useful for your research, please cite our paper

```
wating for ...
```

## Contact

If you have any question, please feel free to contact us. E-mail: [qingsonghu08@gmail.com](mailto:qingsonghu08@gmail.com)
