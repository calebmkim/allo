# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, index
import allo.ir.types as T

import sys

def MATCH(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


def nussinov_np(seq, table):
    N = seq.shape[0]
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i][j] = max(table[i][j], table[i][j - 1])
            if i + 1 < N:
                table[i][j] = max(table[i][j], table[i + 1][j])

            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i][j] = max(
                        table[i][j], table[i + 1][j - 1] + MATCH(seq[i], seq[j])
                    )
                else:
                    table[i][j] = max(table[i][j], table[i + 1][j - 1])

            for k in range(i + 1, j):
                table[i][j] = max(table[i][j], table[i][k] + table[k + 1][j])


def nussinov(concrete_type, n):
    def kernel_nussinov[T: (int32, int32), N: int32](seq: "T[N]", table: "T[N, N]"):
        for i_inv in range(N):
            i: index = N - 1 - i_inv
            for j in range(i + 1, N):
                if j - 1 >= 0:
                    if table[i, j] < table[i, j - 1]:
                        table[i, j] = table[i, j - 1]

                if i + 1 < N:
                    if table[i, j] < table[i + 1, j]:
                        table[i, j] = table[i + 1, j]

                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        w: int32 = seq[i] + seq[j]

                        match: int32 = 0
                        if w == 3:
                            match = 1

                        s2: int32 = 0
                        s2 = table[i + 1, j - 1] + match

                        if table[i, j] < s2:
                            table[i, j] = s2
                    else:
                        if table[i, j] < table[i + 1, j - 1]:
                            table[i, j] = table[i + 1, j - 1]

                for k in range(i + 1, j):
                    s3: int32 = table[i, k] + table[k + 1, j]
                    if table[i, j] < s3:
                        table[i, j] = s3

    s = allo.customize(kernel_nussinov, instantiate=[concrete_type, n])
    return s

def test_nussinov(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["nussinov"][test_psize]["N"]
    concrete_type = int32
    s = nussinov(concrete_type, N)
    if print_hls:
        # printing hls instead
        mod = s.build(target="vhls")
        print(mod)
        return
    mod = s.build()

    seq = np.random.randint(0, 4, size=N).astype(np.int32)
    table = np.zeros((N, N), dtype=np.int32)
    table_ref = table.copy()
    nussinov_np(seq, table_ref)
    mod(seq, table)
    np.testing.assert_allclose(table, table_ref, rtol=1e-5)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_nussinov(print_hls=True)
    else:
        pytest.main([__file__])
