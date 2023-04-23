import mindspore
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Dropout, OneHot, Softmax, Sigmoid
from mindspore.common.initializer import initializer, Normal, Zero, Uniform
import numpy as np


def init_weight(model):
    for _, m in model.cells_and_names():
        if isinstance(m, mindspore.nn.Conv2d):
            out_channel, _, kernel_size_h, kernel_size_w = m.weight.shape
            stddev = np.sqrt(2 / int(out_channel * kernel_size_h * kernel_size_w))
            m.weight.set_data(initializer(Normal(sigma=stddev, mean=0.0),
                                          m.weight.shape,
                                          m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(),
                                            m.bias.shape,
                                            m.bias.dtype))
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out = fan_out // m.groups
            # m.weight = initializer(Normal(sigma=math.sqrt(2.0 / fan_out), mean=0.0), m.weight.shape, dtype=mindspore.float32)
            # if m.bias is not None:
            #     mindspore.common.initializer.Zero(m.bias)
        if isinstance(m, mindspore.nn.Dense):
            init_range = 1.0 / np.sqrt(m.weight.shape[0])
            m.weight.set_data(initializer(Uniform(init_range),
                              m.weight.shape,
                              m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(),
                                            m.bias.shape,
                                            m.bias.dtype))
            # init_range = 1.0 / math.sqrt(1000)
            # uniform = mindspore.common.initializer.Uniform(scale=init_range)
            # uniform(np.array(m.weight))
            # if m.bias is not None:
            #     mindspore.common.initializer.Zero(m.bias)


class Conv(mindspore.nn.Cell):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = mindspore.nn.Conv2d(in_ch, out_ch, k, s, dilation=1, group=g, has_bias=False)
        self.norm = mindspore.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.99)
        self.relu = activation

    def construct(self, x):
        return self.relu(self.norm(self.conv(x)))

class SE(mindspore.nn.Cell):
    def __init__(self, ch, r):
        super().__init__()
        self.se = mindspore.nn.SequentialCell(AdaptiveAvgPool2D(),
                                              mindspore.nn.Conv2d(ch, ch // (4 * r), 1),
                                              mindspore.nn.SiLU(),
                                              mindspore.nn.Conv2d(ch // (4 * r), ch, 1),
                                              mindspore.nn.Sigmoid())

    def construct(self, x):
        return x * self.se(x)


class DropPath(mindspore.nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks). """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = mindspore.nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = mindspore.Tensor(np.ones(shape), dtype=mindspore.dtype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = mindspore.ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self, drop_prob):
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)



class Identity(mindspore.nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = mindspore.ops.Identity()

    def construct(self, x):
        return self.identity(x)

class Residual(mindspore.nn.Cell):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r, dp_rate=0, fused=True):
        super().__init__()
        identity = Identity()
        self.add = s == 1 and in_ch == out_ch

        if fused:
            features = [Conv(in_ch, r * in_ch, activation=mindspore.nn.SiLU(), k=3, s=s),
                        Conv(r * in_ch, out_ch, identity) if r != 1 else identity,
                        DropPath2D(dp_rate) if self.add else identity]
        else:
            features = [Conv(in_ch, r * in_ch, mindspore.nn.SiLU()) if r != 1 else identity,
                        Conv(r * in_ch, r * in_ch, mindspore.nn.SiLU(), 3, s, r * in_ch),
                        SE(r * in_ch, r), Conv(r * in_ch, out_ch, identity),
                        DropPath2D(dp_rate) if self.add else identity]

        self.res = mindspore.nn.SequentialCell(*features)

    def construct(self, x):
        return x + self.res(x) if self.add else self.res(x)

class AdaptiveAvgPool2D(mindspore.nn.Cell):
    def __init__(self):
        super(AdaptiveAvgPool2D, self).__init__()
        self.adaptiveAvgPool2D = mindspore.ops.AdaptiveAvgPool2D(1)

    def construct(self, x):
        return self.adaptiveAvgPool2D(x)

class EfficientNet(mindspore.nn.Cell):
    """
     efficientnet-v2-s :
                        num_dep = [2, 4, 4, 6, 9, 15, 0]
                        filters = [24, 48, 64, 128, 160, 256, 256, 1280]
     efficientnet-v2-m :
                        num_dep = [3, 5, 5, 7, 14, 18, 5]
                        filters = [24, 48, 80, 160, 176, 304, 512, 1280]
     efficientnet-v2-l :
                        num_dep = [4, 7, 7, 10, 19, 25, 7]
                        filters = [32, 64, 96, 192, 224, 384, 640, 1280]
    """

    def __init__(self, drop_rate=0, num_class=1000):
        super().__init__()
        num_dep = [2, 4, 4, 6, 9, 15, 0]
        filters = [24, 48, 64, 128, 160, 256, 256, 1280]

        dp_index = 0
        dp_rates = np.linspace(0, 0.2, sum(num_dep))

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        for i in range(num_dep[0]):
            if i == 0:
                self.p1.append(Conv(3, filters[0], mindspore.nn.SiLU(), 3, 2))
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            else:
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            dp_index += 1
        # p2/4
        for i in range(num_dep[1]):
            if i == 0:
                self.p2.append(Residual(filters[0], filters[1], 2, 4, dp_rates[dp_index]))
            else:
                self.p2.append(Residual(filters[1], filters[1], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p3/8
        for i in range(num_dep[2]):
            if i == 0:
                self.p3.append(Residual(filters[1], filters[2], 2, 4, dp_rates[dp_index]))
            else:
                self.p3.append(Residual(filters[2], filters[2], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p4/16
        for i in range(num_dep[3]):
            if i == 0:
                self.p4.append(Residual(filters[2], filters[3], 2, 4, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[3], filters[3], 1, 4, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[4]):
            if i == 0:
                self.p4.append(Residual(filters[3], filters[4], 1, 6, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[4], filters[4], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        # p5/32
        for i in range(num_dep[5]):
            if i == 0:
                self.p5.append(Residual(filters[4], filters[5], 2, 6, dp_rates[dp_index], False))
            else:
                self.p5.append(Residual(filters[5], filters[5], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[6]):
            if i == 0:
                self.p5.append(Residual(filters[5], filters[6], 2, 6, dp_rates[dp_index], False))
            else:
                self.p5.append(Residual(filters[6], filters[6], 1, 6, dp_rates[dp_index], False))
            dp_index += 1

        self.p1 = mindspore.nn.SequentialCell(*self.p1)
        self.p2 = mindspore.nn.SequentialCell(*self.p2)
        self.p3 = mindspore.nn.SequentialCell(*self.p3)
        self.p4 = mindspore.nn.SequentialCell(*self.p4)
        self.p5 = mindspore.nn.SequentialCell(*self.p5)

        self.fc1 = mindspore.nn.SequentialCell(Conv(filters[6], filters[7], mindspore.nn.SiLU()),
                                               AdaptiveAvgPool2D(),
                                               mindspore.nn.Flatten())
        self.fc2 = mindspore.nn.Dense(filters[7], num_class)

        self.drop_rate = drop_rate

        init_weight(self)

    def construct(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        x = self.fc1(x)
        if self.drop_rate > 0:
            drop = Dropout(1-self.drop_rate)
            x = drop(x)
        return self.fc2(x)

    def export(self):
        for _, m in self.cells_and_names():
            if type(m) is Conv and hasattr(m, 'relu'):
                if isinstance(m.relu, mindspore.nn.SiLU):
                    m.relu = Swish()
            if type(m) is SE:
                if isinstance(m.se[2], mindspore.nn.SiLU):
                    m.se[2] = Swish()
        return self

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.__imul__(Sigmoid(x)) if inplace else x.__mul__(Sigmoid(x))

class Swish(mindspore.nn.Cell):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def construct(self, x):
        return swish(x, self.inplace)

class EMA:
    def __init__(self, model, decay=0.9999):
        super().__init__()
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model)

        self.model.eval()

    def update_fn(self, model, fn):
        for e_std, m_std in zip(self.model.get_parameters(), model.get_parameters()):
            e_std.set_data(fn(e_std, m_std))

    def update(self, model):
        self.update_fn(model, fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(mindspore.nn.Optimizer):
    def __init__(self, params,
                 lr=1e-2, alpha=0.9, eps=1e-3, weight_decay=0, momentum=0.9,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = mindspore.ops.OnesLike(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = mindspore.ops.ZerosLike(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = mindspore.ops.ZerosLike(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class PolyLoss(mindspore.nn.Cell):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    """

    def __init__(self, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon

    def construct(self, outputs, targets):
        ce = SoftmaxCrossEntropyWithLogits(outputs, targets)
        pt = OneHot(targets, outputs.size()[1]) * Softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()


class CrossEntropyLoss(mindspore.nn.Cell):
    """
    NLL Loss with label smoothing.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = mindspore.nn.LogSoftmax(dim=-1)

    def construct(self, x, target):
        prob = self.softmax(x)
        mean = -prob.mean(dim=-1)
        nll_loss = -prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        return ((1. - self.epsilon) * nll_loss + self.epsilon * mean).mean()


# net = EfficientNet()
# net.parameters_and_names()
# for layer in net.cells_and_names():
#     print(layer)
# print(net)