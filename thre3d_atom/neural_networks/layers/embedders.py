import torch

from torch import Tensor
from torch.nn import Module
from typing import Any, Dict


class PositionalEmbeddingsEncoder(Module):
    """
    Embeds input vectors into periodic encodings
    Args:
        input_dims: number of input dimensions
        emb_dims: number of dimensions in the encoded vectors
    """

    def __init__(self, input_dims: int, emb_dims: int):
        super().__init__()
        self._input_dims = input_dims
        self._emb_dims = emb_dims

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "config": {
                "input_dims": self._input_dims,
                "emb_dims": self._emb_dims,
            },
            "state_dict": self.state_dict(),
        }

    @property
    def output_size(self) -> int:
        return self._input_dims + 2 * self._input_dims * self._emb_dims

    def extra_repr(self) -> str:
        return f"input_dims={self._input_dims}, emb_dims={self._emb_dims}"

    def forward(self, x: Tensor) -> Tensor:
        """
        converts the input vectors into positional encodings
        Args:
            x: batch of input_vectors of shape [batch_size x self.input_dims]
        Returns: positional encodings applied input. Shape:
                 [batch_size x (self.input_dims + 2 * self.input_dims * emb_dims)]
                                                  2 for sine and cos
        """
        sin_embedding_dims = (
            torch.arange(0, self._emb_dims, dtype=x.dtype, device=x.device)
            .reshape((1, self._emb_dims))
            .repeat_interleave(repeats=self._input_dims, dim=-1)
        )
        cos_embedding_dims = (
            torch.arange(0, self._emb_dims, dtype=x.dtype, device=x.device)
            .reshape((1, self._emb_dims))
            .repeat_interleave(repeats=self._input_dims, dim=-1)
        )
        for_sines = (2**sin_embedding_dims) * x.repeat(1, self._emb_dims)
        for_coses = (2**cos_embedding_dims) * x.repeat(1, self._emb_dims)
        sines, coses = torch.sin(for_sines), torch.cos(for_coses)
        return torch.cat((x, sines, coses), dim=-1)
