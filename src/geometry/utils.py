# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for geometry library."""

from typing import List

import torch


def unstack(value: torch.Tensor, dim: int = -1) -> List[torch.Tensor]:
    return [torch.squeeze(v, dim=dim) for v in torch.split(value, 1, dim=dim)]

# def unstack(tensor, dim):
#     """
#     Unstack a tensor to list along given dim
#     """
#     shape = tensor.shape
#     return [ tensor.select(dim=dim, index=i) for i in range(shape[dim]) ]
