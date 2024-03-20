# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32
import allo.ir.types as T

import sys

def doitgen_np(A, x, sum):
    NR, NQ, NP = A.shape
    for r in range(NR):
        for q in range(NQ):
            for p in range(NP):
                sum[p] = 0
                for s in range(NP):
                    sum[p] = sum[p] + A[r, q, s] * x[s, p]
            for p in range(NP):
                A[r, q, p] = sum[p]


def doitgen(concrete_type, qq, rr, pp, ss):
    def kernel_doitgen[
        T: (int32, int32), R: int32, Q: int32, P: int32, S: int32
    ](A: "T[R, Q, S]", x: "T[P, S]", sum_: "T[P]"):
        for r, q in allo.grid(R, Q):
            for p in allo.grid(P):
                sum_[p] = 0
                for s in allo.grid(P):
                    sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
            for p1 in allo.grid(P):
                A[r, q, p1] = sum_[p1]

    s0 = allo.customize(kernel_doitgen, instantiate=[concrete_type, rr, qq, pp, ss])
    return s0


def test_doitgen(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    Q = psize["doitgen"][test_psize]["Q"]
    R = psize["doitgen"][test_psize]["R"]
    P = psize["doitgen"][test_psize]["P"]
    S = psize["doitgen"][test_psize]["S"]

    # generate input data
    A = np.random.randint(100, size=(R, Q, S)).astype(np.int32)
    x = np.random.randint(100, size=(P, S)).astype(np.int32)
    sum_ = np.zeros(P).astype(np.int32)
    sum_ref = sum_.copy()

    # run reference
    A_ref = A.copy()
    x_ref = x.copy()
    doitgen_np(A_ref, x_ref, sum_ref)

    # run allo
    A_opt = A.copy()
    x_opt = x.copy()
    s0 =  doitgen(int32, Q, R, P, S)
    if print_hls:
        # printing hls instead
        mod = s0.build(target="vhls")
        print(mod)
        return
    mod = s0.build()
    mod(A_opt, x_opt, sum_)

    # compare
    np.testing.assert_allclose(A_ref, A_opt)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_doitgen(print_hls=True)
    else:
        pytest.main([__file__])
