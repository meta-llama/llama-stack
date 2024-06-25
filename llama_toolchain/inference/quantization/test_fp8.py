# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import unittest

import torch

from fp8_impls import ffn_swiglu_fp8_dynamic, FfnQuantizeMode, quantize_fp8
from hypothesis import given, settings, strategies as st
from torch import Tensor


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class FP8Tests(unittest.TestCase):
    @settings(deadline=None)
    @given(
        D=st.sampled_from([4096, 8192]),
        HD_L=st.sampled_from([1280, 2560]),
        B=st.sampled_from([1, 2]),
        T=st.sampled_from([2048, 4096]),
        UB=st.sampled_from([1000, 10000]),
    )
    def test_fp8_ffn(
        self,
        D: int,  # noqa
        HD_L: int,
        B: int,
        T: int,
        UB: float,
    ) -> None:
        x = torch.randn(size=(B, T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w1 = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01
        w3 = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01
        w2 = torch.randn(size=(D, HD_L), dtype=torch.bfloat16, device="cuda") * 0.1

        x_q = quantize_fp8(x, UB, mode=FfnQuantizeMode.FP8_ROWWISE)
        w1_q = quantize_fp8(w1, UB, mode=FfnQuantizeMode.FP8_ROWWISE)
        w3_q = quantize_fp8(w3, UB, mode=FfnQuantizeMode.FP8_ROWWISE)
        w2_q = quantize_fp8(w2, UB, mode=FfnQuantizeMode.FP8_ROWWISE)

        def ref_ffn(x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor) -> Tensor:
            (B, T, D) = x.shape  # noqa: N806
            (HD_L, D_) = w1.shape  # noqa: N806
            assert D_ == D

            x1 = x.view(B * T, D) @ w1.T
            x2 = x.view(B * T, D) @ w3.T

            z = torch.nn.functional.silu(x1) * x2
            return (z @ w2.T).view(B, T, D).to(torch.bfloat16)

        v = ffn_swiglu_fp8_dynamic(x, w1_q, w3_q, w2_q)

        # Fake quant
        x = x_q.weight.bfloat16() * x_q.scale.unsqueeze(-1)
        w1 = w1_q.weight.bfloat16() * w1_q.scale.unsqueeze(-1)
        w3 = w3_q.weight.bfloat16() * w3_q.scale.unsqueeze(-1)
        w2 = w2_q.weight.bfloat16() * w2_q.scale.unsqueeze(-1)

        v_ref = ref_ffn(x, w1, w3, w2)

        torch.testing.assert_close(v_ref, v, atol=4.0e-3, rtol=4.0e-3)


if __name__ == "__main__":
    unittest.main()
