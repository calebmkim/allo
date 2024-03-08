# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def atax(concrete_type, m, n):
    def stage_M[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", out_Ax: "T[M]"):
        for m in allo.grid(M):
            for r in allo.reduction(N):
                out_Ax[m] += A[m, r] * x[r]

    def stage_N[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", out_Ax: "T[M]", y: "T[N]"):
        for n in allo.grid(N):
            for k in allo.reduction(M):
                y[n] += A[k, n] * out_Ax[k]

    def kernel_atax[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", y: "T[N]"):
        out_Ax: T[M] = 0
        stage_M[T, M, N](A, x, out_Ax)
        stage_N[T, M, N](A, out_Ax, y)

    sch0 = allo.customize(stage_M, instantiate=[concrete_type, m, n])
    sch0.reorder("r", "m")
    sch0.pipeline("m")
    # unroll factor 39

    sch1 = allo.customize(stage_N, instantiate=[concrete_type, m, n])
    sch1.reorder("k", "n")
    sch1.pipeline("n")
    # unroll factor 41

    sch = allo.customize(kernel_atax, instantiate=[concrete_type, m, n])
    sch.compose(sch0)
    sch.compose(sch1)

    code = sch.build(target="vhls")
    print(code)


if __name__ == "__main__":
    atax()
