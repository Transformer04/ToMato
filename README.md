# ToMato: Token Merging at Once
![ToMato](https://github.com/Transformer04/ToMato/assets/81407149/87c51941-4da8-4fdd-a2a1-143ed639addc)
## About our code
ViT(Vision Transformer) shows outstanding performance in various vision tasks by splitting images into patches and passing them through transformer blocks. However, the large model size and computational cost of ViT result in high inference latency and hindered acceleration. To accelerate ViT efficiently, we introduce ToMato(Token Merging at Once), a simple framework that recursively merges tokens by comparing similarity to adjacent tokens at the first transformer block. Applying the ToMato to DeiT-base model, we find that this reduces latency by 22.19% while maintaining high Top-1 accuracy of 80.14%.

## How to install
git clone our repository to your computer

## How to test
If you want to evaluate the accuracy of our model, enter <test_batch.py> file and change the directory path to your dataset in line 40.
Then, run test_batch.py

```
python test_batch.py
```

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

![Visualization](https://github.com/Transformer04/ToMato/assets/81407149/d593a9ce-b0da-4af5-bbe2-c5231e905617)

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
