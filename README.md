# `isaac` Newton - Accelerating NN Training with Input-based Approximate Curvature for Newton's Method

<img align="right" height="150" src="isaac_newton_logo.png" />

This repository includes the official implementation of our ICLR 2023 Paper "ISAAC Newton: Input-based Approximate Curvature for Newton's Method".

Paper @ [OpenReview](https://openreview.net/pdf?id=0paCJSFW7j) 

[//]: # (/ [ArXiv]&#40;https://arxiv.org/pdf/.pdf&#41;)

Video @ [Youtube](https://youtu.be/7RKRX-MdwqM)

[![video](https://www.petersen.ai/images/isaac_title_slide_small.jpg)](https://youtu.be/7RKRX-MdwqM)

## 💻 Installation

`isaac` is based on PyTorch and can be installed via pip from PyPI with
```shell
pip install isaac
```

## 👩‍💻 Usage

`isaac.Linear` acts as a drop-in replacement for `torch.nn.Linear`. It only requires additional specification of
the regularization parameter $\lambda_a$ `la` as specified in the paper. A good starting point for $\lambda_a$
is `la=1`, but the optimal choice varies from experiment to experiment.
The method operates by efficiently modifying the gradient of the module in such a way that the input-based curvature
information is used when applying a gradient descent optimizer on the modified gradients. 

In the following, we specify an example MNIST neural network where ISAAC is applied to the first 3 out of 5 layers:

```python
import torch
import isaac

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    isaac.Linear(784, 1_600, la=1),
    torch.nn.ReLU(),
    isaac.Linear(1_600, 1_600, la=1),
    torch.nn.ReLU(),
    isaac.Linear(1_600, 1_600, la=1),
    torch.nn.ReLU(),
    torch.nn.Linear(1_600, 1_600),
    torch.nn.ReLU(),
    torch.nn.Linear(1_600, 10)
)
```

## 🧪 Experiments 

You can find an example MNIST experiment in `examples/mnist.py`, which is based on the experiment in Figure 5 in the paper.

To run ISAAC applied to the first `X` out of 5 layers, run
```shell
python examples/mnist.py -nil <X>
```

To run the baseline, run
```shell
python examples/mnist.py -nil 0
```

The device can be specified, e.g., as `--device cuda`, the learning rate and $\lambda_a$ may be set via `--lr` and `--la`, respectively.

## 📖 Citing

```bibtex
@inproceedings{petersen2023isaac,
  title={ISAAC Newton: Input-based Approximate Curvature for Newton's Method},
  author={Petersen, Felix and Sutter, Tobias and Borgelt, Christian and Huh, Dongsung and Kuehne, Hilde and Sun, Yuekai and Deussen, Oliver},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

`isaac` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
