from typing import Callable, Sequence, Any, Optional, Tuple, List

import numpy as np
from tqdm import tqdm


def check_power_of_2(x: int) -> bool:
    return x & (x - 1) == 0


def batchify(
    processor_fn: Callable[[Sequence[Any], Any], Sequence[Any]],
    collate_fn: Callable[[Sequence[Any]], Any],
    chunk_size: Optional[int] = None,
    verbose: bool = False,
) -> Callable[[Sequence[Any]], Sequence[Any]]:
    if chunk_size is None:
        return processor_fn

    def batchified_processor_fn(inputs: Sequence[Any], *args, **kwargs) -> Any:
        processed_inputs_batches = []
        progress_bar = tqdm if verbose else lambda _: _
        for chunk_index in progress_bar(range(0, len(inputs), chunk_size)):
            processed_inputs_batches.append(
                processor_fn(
                    inputs[chunk_index : chunk_index + chunk_size], *args, **kwargs
                )
            )

        return collate_fn(processed_inputs_batches)

    return batchified_processor_fn


def compute_thre3d_grid_sizes(
    final_required_resolution: Tuple[int, int, int],
    num_stages: int,
    scale_factor: float,
) -> List[Tuple[int, int, int]]:
    x, y, z = final_required_resolution
    grid_sizes = [(x, y, z)]
    for _ in range(num_stages - 1):
        x = int(np.ceil((1 / scale_factor) * x))
        y = int(np.ceil((1 / scale_factor) * y))
        z = int(np.ceil((1 / scale_factor) * z))
        grid_sizes.insert(0, (x, y, z))
    return grid_sizes
