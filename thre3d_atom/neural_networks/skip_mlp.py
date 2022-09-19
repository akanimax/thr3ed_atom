import torch

from typing import Sequence
from torch import Tensor
from torch.nn import (
    LeakyReLU,
    Identity,
    Linear,
    Module,
    ModuleList,
    Sequential,
    Dropout,
)


class SkipMLP(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        layer_depths: Sequence[int] = (128, 128),
        skips: Sequence[bool] = (False, False),
        dropout_prob: float = 0.0,
        activation_fn: Module = LeakyReLU(),
        out_activation_fn: Module = Identity(),
    ) -> None:
        super().__init__()

        # define the state of the SkipMLP object
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._layer_depths = layer_depths
        self._skips = skips
        self._dropout_prob = dropout_prob
        self._activation_fn = activation_fn
        self._out_activation_fn = out_activation_fn

        self.network_layers = self._get_sequential_layers_with_skips()

        # initialize the layers according to the glorot_uniform
        self.apply(self._init_weights_glorot_uniform)

    @staticmethod
    def _init_weights_glorot_uniform(module: Module) -> None:
        if type(module) == Linear:
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.fill_(0.0)

    def _get_sequential_layers_with_skips(self) -> ModuleList:
        modules, in_features = [], self._input_dims
        for skip, layer_depth in zip(self._skips, self._layer_depths):
            modules.append(
                Sequential(
                    Linear(
                        in_features=in_features,
                        out_features=layer_depth,
                        bias=True,
                    ),
                    self._activation_fn,
                    Dropout(self._dropout_prob),
                )
            )
            in_features = layer_depth + self._input_dims if skip else layer_depth

        # add the final output layer separately:
        modules.append(Linear(in_features, self._output_dims, bias=True))
        modules.append(self._out_activation_fn)
        return ModuleList(modules)

    def get_save_info(self):
        return {
            "config": {
                "input_dims": self._input_dims,
                "output_dims": self._output_dims,
                "layer_depths": self._layer_depths,
                "skips": self._skips,
                "dropout_prob": self._dropout_prob,
                "activation_fn": self._activation_fn,
                "out_activation_fn": self._out_activation_fn,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tensor:
        y = self.network_layers[0](x)  # first layer doesn't have skips
        for skip, network_layer in zip(self._skips, self.network_layers[1:]):
            y = torch.cat([y, x], dim=-1) if skip else y
            y = network_layer(y)
        return self.network_layers[-1](y)
