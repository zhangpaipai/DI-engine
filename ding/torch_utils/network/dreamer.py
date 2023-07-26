import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from ding.torch_utils import MLP

class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class DreamerLayerNorm(nn.Module):

    def __init__(self, ch, eps=1e-03):
        super(DreamerLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DenseHead(nn.Module):

    def __init__(
        self,
        inp_dim,  # config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        shape,  # (255,)
        layer_num,
        units,  # 512
        act='SiLU',
        norm='LN',
        dist='normal',
        std=1.0,
        outscale=1.0,
        device='cpu',
    ):
        super(DenseHead, self).__init__()
        self._shape = (shape, ) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = (1, )
        self._layer_num = layer_num
        self._units = units
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, 'LayerNorm')
        #self._act = getattr(torch.nn, act)()
        #self._norm = norm
        self._dist = dist
        self._std = std
        self._device = device

        # self.mlp = MLP(
        #     inp_dim,
        #     self._units,
        #     self._units,
        #     self._layer_num,
        #     layer_fn=nn.Linear,
        #     activation=self._act,
        #     norm_type=self._norm
        # )
        layers = []
        for index in range(self._layer_num):
            layers.append(nn.Linear(inp_dim, self._units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

        self.mean_layer = nn.Linear(self._units, np.prod(self._shape))
        self.mean_layer.apply(uniform_weight_init(outscale))

        if self._std == "learned":
            self.std_layer = nn.Linear(self._units, np.prod(self._shape))
            self.std_layer.apply(uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        out = self.layers(x)  # (batch, time, _units=512)
        mean = self.mean_layer(out)  # (batch, time, 255)
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        if self._dist == "normal":
            return ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
        if self._dist == "huber":
            return ContDist(torchd.independent.Independent(UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
        if self._dist == "binary":
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
        if self._dist == "twohot_symlog":
            return TwoHotDistSymlog(logits=mean, device=self._device)
        raise NotImplementedError(self._dist)


class ActionHead(nn.Module):

    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = getattr(torch.nn, act)
        self._norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(self._norm(self._units, eps=1e-03))
            pre_layers.append(self._act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

    def __call__(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist
    

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):

    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class TwoHotDistSymlog():

    def __init__(self, logits=None, low=-20.0, high=20.0, device='cpu'):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255

    def mean(self):
        print("mean called")
        _mean = self.probs * self.buckets
        return symexp(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return symexp(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = symlog(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum((self.buckets > x[..., None]).to(torch.int32), dim=-1)
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = (below == above)

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None] +
            F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class SymlogDist():

    def __init__(self, mode, dist='mse', agg='sum', tol=1e-8, dim_to_reduce=[-1, -2, -3]):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._dim_to_reduce = dim_to_reduce

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == 'mse':
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == 'abs':
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == 'mean':
            loss = distance.mean(self._dim_to_reduce)
        elif self._agg == 'sum':
            loss = distance.sum(self._dim_to_reduce)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):

    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2) - self._threshold)

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where((torch.abs(y) <= 1.), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan(fn, inputs, start):
    last = start  # {logit:[batch_size, self._stoch, self._discrete], stoch:[batch_size, self._stoch, self._discrete], deter:[batch_size, self._deter]}
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)  # inputs:(action:(time, batch, 6), embed:(time, batch, 4096))
        last = fn(last, *inp(index))  # post, prior
        if flag:
            if type(last) == type({}):
                outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat([outputs[j][key], last[j][key].unsqueeze(0)], dim=0)
                    else:
                        outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):

    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    return f

class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy().item()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics
    
    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


class ConvEncoder(nn.Module):
    def __init__(
        self,
        grayscale=False,
        depth=32,
        act=nn.ELU,
        norm=nn.LayerNorm,
        kernels=(3, 3, 3, 3),
    ):
        super(ConvEncoder, self).__init__()
        self._act = act
        self._norm = norm
        self._depth = depth
        self._kernels = kernels
        h, w = 64, 64
        layers = []
        for i, kernel in enumerate(self._kernels):
            if i == 0:
                if grayscale:
                    inp_dim = 1
                else:
                    inp_dim = 3
            else:
                inp_dim = 2 ** (i - 1) * self._depth
            depth = 2**i * self._depth
            layers.append(
                Conv2dSame(
                    in_channels=inp_dim,
                    out_channels=depth,
                    kernel_size=(kernel, kernel),
                    stride=(2, 2),
                    bias=False,
                )
            )
            layers.append(DreamerLayerNorm(depth))
            layers.append(act())
            h, w = h // 2, w // 2

        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

    def __call__(self, obs):
        #x = obs["image"].reshape((-1,) + tuple(obs["image"].shape[-3:]))
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        #x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # prod: product of all elements
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        #shape = list(obs.shape[:-3]) + [x.shape[-1]]
        #return x.reshape(shape)
        return x