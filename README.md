# Distributed DVC

The official PyTorch implementation of our **ICME 2023** paper: 

**Low-complexity Deep Video Compression with A Distributed Coding Architecture**

[Xinjie Zhang](https://xinjie-q.github.io/), [Jiawei Shao](https://shaojiawei07.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/)

[[ArXiv Preprint](https://arxiv.org/abs/2303.11599)]]

### :bookmark:Brief Introduction

Prevalent predictive coding-based video compression methods rely on a heavy encoder to reduce temporal redundancy, which makes it challenging to deploy them on resource-constrained devices. Since the 1970s, distributed source coding theory has indicated that independent encoding and joint decoding with side information (SI) can achieve high-efficient compression of correlated sources. This has inspired a *distributed coding* architecture aiming at reducing the encoding complexity. However, traditional distributed coding methods suffer from a substantial performance gap to predictive coding ones. Inspired by the great success of learning-based compression, we propose the first end-to-end distributed deep video compression framework to improve the rate-distortion performance. A key ingredient is an effective SI generation module at the decoder, which helps to effectively exploit inter-frame correlations without computation-intensive encoder-side motion estimation and compensation. Experiments show that our method significantly outperforms conventional distributed video coding and H.264. Meanwhile, it enjoys 6~7x encoding speedup against DVC with comparable compression performance. 

## Acknowledgement

:heart::heart::heart:Our idea is implemented based on the following projects. We really appreciate their wonderful open-source works!

- [CompressAI](https://github.com/InterDigitalInc/CompressAI) [[related paper](https://arxiv.org/abs/2011.03029)]
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) [[related paper](https://arxiv.org/abs/2011.06294)]

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@inproceedings{zhang2023low,
  title={Low-complexity Deep Video Compression with A Distributed Coding Architecture},
  author={Zhang, Xinjie and Shao, Jiawei and Zhang, Jun},
  booktitle={IEEE International Conference on Multimedia and Expo},
  year={2023},
}

@article{zhang2023low,
  title={Low-complexity Deep Video Compression with A Distributed Coding Architecture},
  author={Zhang, Xinjie and Shao, Jiawei and Zhang, Jun},
  journal={arXiv preprint arXiv:2303.11599},
  year={2023}
}
```

