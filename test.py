import torch
from torch.autograd import gradcheck
import torch.nn.functional as F
from torch.autograd import gradcheck
import math
import torch.nn as nn
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)
batch_size = 1
input_feature = 4
hidden_feature = 8
out_feature = 2
# Issue 2:
real = ti.f64
ti_data = ti.var(dt=real, shape=(batch_size, input_feature), needs_grad=True)
ti_weight_0 = ti.var(dt=real, shape=(input_feature, hidden_feature), needs_grad=True)
ti_bias_0 = ti.var(dt=real, shape=hidden_feature, needs_grad=True)
ti_output_0 = ti.var(dt=real, shape=(batch_size, hidden_feature), needs_grad=True)
ti_weight_1 = ti.var(dt=real, shape=(hidden_feature, out_feature), needs_grad=True)
ti_bias_1 = ti.var(dt=real, shape=out_feature, needs_grad=True)
# Issue 1:
ti_output_1 = ti.var(dt=real, shape=(batch_size, out_feature), needs_grad=True)


@ti.kernel
def linear_kernel():
    for i in range(batch_size):
        for j in ti.static(range(hidden_feature)):
            dummy = 0.0
            for k in ti.static(range(input_feature)):
                dummy += ti_data[i, k] * ti_weight_0[k, j]
            dummy += ti_bias_0[j]
            ti_output_0[i, j] = ti.max(dummy, 0)
        for j in ti.static(range(out_feature)):
            dummy = 0.0
            for k in ti.static(range(hidden_feature)):
                dummy += ti_output_0[i, k] * ti_weight_1[k, j]
            dummy += ti_bias_1[j]
            ti_output_1[i, j] = dummy


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, weight_0, bias_0, weight_1, bias_1, bias=None):
        ti_data.from_torch(input_data)
        ti_weight_0.from_torch(weight_0)
        ti_bias_0.from_torch(bias_0)
        ti_weight_1.from_torch(weight_1)
        ti_bias_1.from_torch(bias_1)
        linear_kernel()
        return ti_output_1.to_torch()

    @staticmethod
    def backward(ctx, grad_output_1):
        ti.clear_all_gradients()
        grad_input_data = grad_weight_0 = grad_bias_0 = grad_weight_1 = grad_bias_1 = None
        ti_output_1.grad.from_torch(grad_output_1)
        linear_kernel.grad()

        if ctx.needs_input_grad[0]:
            grad_input_data = ti_data.grad.to_torch()
        if ctx.needs_input_grad[1]:
            grad_weight_0 = ti_weight_0.grad.to_torch()
        if ctx.needs_input_grad[2]:
            grad_bias_0 = ti_bias_0.grad.to_torch()
        if ctx.needs_input_grad[3]:
            grad_weight_1 = ti_weight_1.grad.to_torch()
        if ctx.needs_input_grad[4]:
            grad_bias_1 = ti_bias_1.grad.to_torch()

        return grad_input_data, grad_weight_0, grad_bias_0, grad_weight_1, grad_bias_1


class Linear(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature):
        super(Linear, self).__init__()
        self.weight_0 = nn.Parameter(torch.Tensor(input_feature, hidden_feature))
        self.bias_0 = nn.Parameter(torch.Tensor(hidden_feature))
        self.weight_1 = nn.Parameter(torch.Tensor(hidden_feature, out_feature))
        self.bias_1 = nn.Parameter(torch.Tensor(out_feature))
        self.weight_0.data.normal_(0, math.sqrt(2. / hidden_feature / input_feature))
        self.weight_1.data.normal_(0, math.sqrt(2. / hidden_feature / output_feature))

    def forward(self, input_data):
        return LinearFunction.apply(input_data, self.weight_0, self.bias_0, self.weight_1, self.bias_1)


data = torch.rand(batch_size, input_feature, dtype=torch.float64, requires_grad=True)
linear = Linear(input_feature, hidden_feature, out_feature).double()

test = gradcheck(linear, data, eps=1e-6, atol=1e-4)
print(test)