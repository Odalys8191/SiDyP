import torch
import math
import torch.nn.functional as F
from collections.abc import Mapping

def scale(inputs, scale_value):
    return inputs / scale_value

def convert_to_simplex(label, simplex_value, num_classes):
    # 如果label是整数标签，使用one-hot编码
    if label.dtype in [torch.int32, torch.int64] and label.dim() == 1:
        return 2 * simplex_value * F.one_hot(label, num_classes) - simplex_value
    # 如果label是连续标签分布，直接缩放
    else:
        return 2 * simplex_value * label - simplex_value

def logits_projection(logits, simplex_value):
    probs = F.softmax(logits, dim=-1)
    # 直接使用连续概率分布，不再转换为离散标签
    return convert_to_simplex(probs, simplex_value, logits.size(-1))

def self_condition_preds(self_condition, logits, logits_projection=None):
    if self_condition in ["logits", "logits_addition", "logits_mean", "logits_max", "logits_multiply"]:
        previous_pred = logits.detach()
    elif self_condition in ["logits_with_projection", "logits_with_projection_addition"]:
        previous_pred = logits_projection(logits.detach())
    else:
        assert NotImplementedError(f"{self_condition} is not implemented.")
    return previous_pred

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


def mix_values_based_on_self_condition(self_condition_type, value_1, value_2):
    if self_condition_type in ["logits_with_projection_addition", "logits_addition"]:
        mixed_values = value_1 + value_2
    elif self_condition_type == "logits_mean":
        mixed_values = (value_1 + value_2) / 2.0
    elif self_condition_type == "logits_max":
        mixed_values = torch.max(value_1, value_2)
    elif self_condition_type == "logits_multiply":
        mixed_values = value_1 * value_2
    else:
        assert NotImplementedError
    return mixed_values


def adjust_learning_rate(optimizer, epoch, warmup_epochs=100, n_epochs=1000, lr_input=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
