import importlib
from typing import Type

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def get_activation(activation_f: str) -> Type:
    package_name = "mlx.nn.layers.activations"
    module = importlib.import_module(package_name)

    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [
        cls
        for cls in activations
        if isinstance(cls, type) and issubclass(cls, nn.Module)
    ]
    names = [cls.__name__.lower() for cls in activations]

    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(
            f"get_activation: {activation_f=} is not yet implemented."
        )


class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @property
    def num_params(self):
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f"Number of parameters: {self.num_params}")

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Subclass must implement this method")
