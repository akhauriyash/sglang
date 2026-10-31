# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measured GEMM row-count policy for native Gekko FFN down projections.

The native Gekko oracle pads token prefixes to 4096. For the qualified SM90
BF16 5120-to-1536 down projection, smaller GEMMs can round differently. Padding
short calls to 256 rows reproduced the native learned projections and tested
model traces. This is a policy for that configuration, not a general guarantee
about cuBLAS algorithms, arbitrary checkpoints, or other GPU architectures.

At M=1 this computes 256 output rows and discards 255. It also allocates a
256-by-5120 BF16 input buffer (2.5 MiB); throughput must be measured separately.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GekkoDownProjection(nn.Linear):
    """Preserve native parameter names while stabilizing short Gekko GEMMs."""

    minimum_rows = 256

    def forward(self, input: Tensor) -> Tensor:
        if (
            input.ndim == 0
            or input.numel() == 0
            or input.shape[-1] != 5120
            or self.weight.shape != (1536, 5120)
            or self.bias is not None
            or input.dtype != torch.bfloat16
            or self.weight.dtype != torch.bfloat16
            or not input.is_cuda
            or self.weight.device != input.device
            or torch.version.hip is not None
            or torch.is_autocast_enabled("cuda")
            or (
                torch.is_grad_enabled()
                and (input.requires_grad or self.weight.requires_grad)
            )
        ):
            return F.linear(input, self.weight, self.bias)
        rows = input.numel() // input.shape[-1]
        if rows >= self.minimum_rows or torch.cuda.get_device_capability(
            input.device
        ) != (9, 0):
            return F.linear(input, self.weight, self.bias)
        flattened = input.reshape(rows, self.in_features)
        padded = F.pad(flattened, (0, 0, 0, self.minimum_rows - rows))
        output = F.linear(padded, self.weight, self.bias)[:rows]
        return output.reshape(*input.shape[:-1], self.out_features)
