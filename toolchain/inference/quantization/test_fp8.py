# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import unittest

import torch

from fp8_impls import attn_linear, ffn_swiglu_fp8_dynamic, quantize_fp8
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
        D: int,
        HD_L: int,
        B: int,
        T: int,
        UB: float,
    ) -> None:
        x = torch.randn(size=(B, T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w13 = (
            torch.randn(size=(2 * HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01
        )
        w2 = torch.randn(size=(D, HD_L), dtype=torch.bfloat16, device="cuda") * 0.1

        x_q = quantize_fp8(x, UB)
        w13_q = quantize_fp8(w13, UB)
        w2_q = quantize_fp8(w2, UB)

        def ref_ffn(x: Tensor, w13: Tensor, w2: Tensor) -> Tensor:
            (B, T, D) = x.shape
            (HD_L_2, D_) = w13.shape
            assert D_ == D
            HD_L = HD_L_2 // 2

            y = x.view(B * T, D) @ w13.T
            x1 = y[:, :HD_L]
            x2 = y[:, HD_L:]

            z = torch.nn.functional.silu(x1) * x2
            return (z @ w2.T).view(B, T, D).to(torch.bfloat16)

        v = ffn_swiglu_fp8_dynamic(x, w13_q, w2_q)

        # Fake quant
        x = x_q.weight.bfloat16() * x_q.scale
        w13 = w13_q.weight.bfloat16() * w13_q.scale
        w2 = w2_q.weight.bfloat16() * w2_q.scale

        v_ref = ref_ffn(x, w13, w2)

        torch.testing.assert_close(v_ref, v, atol=4.0e-3, rtol=4.0e-3)

    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        UB=st.sampled_from([1000, 10000]),
    )
    def test_fp8_attn_linear(self, B_T: int, D: int, HD_L: int, UB: int) -> None:
        B_T = 4096
        D = 256
        HD_L = 512
        UB = float(UB)
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        wqkv = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01

        x_q = quantize_fp8(x, UB)
        wqkv_q = quantize_fp8(wqkv, UB)

        num_tokens = torch.tensor(B_T, dtype=torch.int64, device="cuda")

        y = attn_linear(x, wqkv_q)
        y_nt = attn_linear(x, wqkv_q, num_tokens=num_tokens)

        # Fake quant
        x = x_q.weight.bfloat16() * x_q.scale
        wqkv = wqkv_q.weight.bfloat16() * wqkv_q.scale
        y_ref = (x @ wqkv.T).to(torch.bfloat16)

        torch.testing.assert_close(y_ref, y, atol=1.0e-3, rtol=1.0e-3)
        torch.testing.assert_close(y_ref, y_nt, atol=1.0e-3, rtol=1.0e-3)


if __name__ == "__main__":
    unittest.main()
