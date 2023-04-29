"""
isaac
Copyright (c) Felix Petersen.
This source code is licensed under the MIT license found in the LICENSE file.
"""


import torch


class ISAACLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, la, inv_type):
        ctx.save_for_backward(input, weight, bias)
        ctx.la = la
        if inv_type == 'cholesky_inverse':
            ctx.inverse = torch.cholesky_inverse
        elif inv_type == 'inverse':
            ctx.inverse = torch.inverse
        else:
            raise NotImplementedError(inv_type)
        return input @ weight.T + (bias if bias is not None else 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        assert len(input.shape) == 2, input.shape
        assert len(grad_output.shape) == 2, grad_output.shape

        if ctx.needs_input_grad[0]:
            grad_0 = grad_output @ weight
        else:
            grad_0 = None

        if ctx.needs_input_grad[1]:
            if input.shape[0] < input.shape[1]:
                aaT = input @ input.T / input.shape[0]
                I_b = torch.eye(aaT.shape[0], device=aaT.device, dtype=aaT.dtype)
                aaT_IaaT_inv = aaT @ ctx.inverse(aaT / ctx.la + I_b)
                grad_1 = grad_output.T @ (
                        I_b - 1. / ctx.la * aaT_IaaT_inv
                ) @ input

            else:
                aTa = input.T @ input / input.shape[0]
                I_n = torch.eye(aTa.shape[0], device=aTa.device, dtype=aTa.dtype)
                grad_1 = grad_output.T @ input @ ctx.inverse(aTa + ctx.la * I_n) * ctx.la

        else:
            grad_1 = None

        return (
            grad_0,
            grad_1,
            grad_output.sum(0, keepdim=True) if bias is not None else None,
            None, None
        )


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features,
                 la, inv_type='inverse', **kwargs):
        super(Linear, self).__init__(
            in_features=in_features, out_features=out_features, **kwargs
        )
        self.la = la
        self.inv_type = inv_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ISAACLinearFunction.apply(
            input, self.weight,
            self.bias.unsqueeze(0) if self.bias is not None else None,
            self.la,
            self.inv_type
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, la={}, bias={}'.format(
            self.in_features, self.out_features, self.la, self.bias is not None
        )


########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    torch.manual_seed(0)

    x = torch.randn(30, 8)
    y = torch.randn(30, 5)
    lin = Linear(8, 5, 0.1)
    y_hat = lin(x)
    loss = (y - y_hat).pow(2).mean()
    loss.backward()

    print(lin.weight.grad)
    print(lin.bias.grad)

    """
tensor([[ 0.0074,  0.0095, -0.0056, -0.0040, -0.0136,  0.0001,  0.0160, -0.0047],
        [-0.0099, -0.0106, -0.0178,  0.0148,  0.0111, -0.0037,  0.0032, -0.0133],
        [ 0.0030,  0.0048,  0.0138, -0.0168,  0.0078, -0.0085, -0.0059,  0.0053],
        [-0.0073,  0.0105,  0.0100,  0.0042, -0.0141, -0.0003, -0.0043, -0.0115],
        [ 0.0144, -0.0073, -0.0026, -0.0101, -0.0090,  0.0182, -0.0152,  0.0095]])
tensor([-0.1162,  0.1062, -0.1275, -0.0206, -0.1149])
    """

    ####################################################################################################################

    import matplotlib.pyplot as plt
    import time
    import tqdm
    import numpy as np

    bss = list(range(1, 2001, 20))
    num_it = 30

    times = []
    lin = Linear(1_000, 5_000, 0.1)
    plt.title(lin)
    for i in tqdm.tqdm(bss):
        x = torch.randn(i, 1_000)
        ts = []
        for _ in range(num_it):
            t_s = time.time()
            y_hat = lin(x)
            ts.append(time.time() - t_s)
        times.append(np.median(ts) * 1000)

    plt.plot(bss, times, label='Linear')

    times = []
    for i in tqdm.tqdm(bss):
        x = torch.randn(i, 1_000)
        y = torch.randn(i, 5_000)
        ts = []
        for _ in range(num_it):
            t_s = time.time()
            y_hat = lin(x)
            loss = (y - y_hat).pow(2).mean()
            loss.backward()
            ts.append(time.time() - t_s)
        times.append(np.median(ts) * 1000)

    plt.plot(bss, times, label='Linear with backward')
    plt.legend()
    plt.ylabel('[ms]')
    plt.xlabel('bs')
    plt.show()

