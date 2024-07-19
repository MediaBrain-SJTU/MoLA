## Exploring Training on Heterogeneous Data with Mixture of Low-rank Adapters


This repository is an official PyTorch implementation of "Exploring Training on Heterogeneous Data with Mixture of Low-rank Adapters"

### Download Dataset

- VLCS  [here](https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file).
- OfficeHOME  [here](https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip).
- RadImageNet  [here](https://github.com/BMEII-AI/RadImageNet).
- MedMNIST  [here](https://github.com/MedMNIST/MedMNIST).
- NYUv2  [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0).

### Run


For example, you can run the following command for MoLA training.

```shell
python train.py --benchmark vlcs_resnet50 --balancer ourbase --arch lora_soft_router --lora_layer 1 2 3 --lora_rank 4 4 4 8
```


## Citation

If you find ``MoLA`` useful for your research or development, please cite the following:

```latex
@article{zhou2024exploring,
  title={Exploring Training on Heterogeneous Data with Mixture of Low-rank Adapters},
  author={Zhou, Yuhang and Zhao, Zihua and Li, Haolin and Du, Siyuan and Yao, Jiangchao and Zhang, Ya and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2406.09679},
  year={2024}
}
```



## Acknowledgement

This repository is built based on [LibMTL](https://github.com/median-research-group/LibMTL). We thank the authors for releasing their codes.
