# ToMato: Token Merging at Once

## About our code


## How to build


## How to install


## How to test


## Datasets
Test and validation were conducted using the Imagenet-mini-1000 dataset. The dataset can be checked at the following link.
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000


## Experiment Results
Here are some expected results when using the timm implementation *off-the-shelf* on ImageNet-1k val using a V100:

| Model        | Top-1 acc (%) | Top-5 acc (%) |  Latency (s)|
|--------------|--------------:|--------------:|:-----------:|
| DeiT-B       |        81.41  |           953 |     13.2132 |
| ToMe-B       |        84.57  |           309 |          13 |
| OURS-B       |        85.82  |            95 |           7 |



## License and Contributing

This code has been implemented with reference to ToMe's code.
Official PyTorch implemention of **ToMe** from the paper: [Token Merging: Your ViT but Faster](https://arxiv.org/abs/2210.09461).  
Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.

Please refer to the [CC-BY-NC 4.0](LICENSE). For contributing, see [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

```
@inproceedings{bolya2022tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
