# ToMato: Token Merging at Once

This code has been implemented with reference to ToMe's code.
Official PyTorch implemention of **ToMe** from the paper: [Token Merging: Your ViT but Faster](https://arxiv.org/abs/2210.09461).  
Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.

Here are some expected results when using the timm implementation *off-the-shelf* on ImageNet-1k val using a V100:

| Model          | original acc | original im/s |  r | ToMe acc | ToMe im/s |
|----------------|-------------:|--------------:|:--:|---------:|----------:|
| ViT-S/16       |        81.41 |           953 | 13 |    79.30 |      1564 |
| ViT-B/16       |        84.57 |           309 | 13 |    82.60 |       511 |
| ViT-L/16       |        85.82 |            95 |  7 |    84.26 |       167 |
| ViT-L/16 @ 384 |        86.92 |            28 | 23 |    86.14 |        56 |


## License and Contributing

Please refer to the [CC-BY-NC 4.0](LICENSE). For contributing, see [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation
If you use ToMe or this repository in your work, please cite:
```
@inproceedings{bolya2022tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
