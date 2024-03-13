# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, int128, index, UInt
from allo.ir.utils import MockBuffer
from allo.utils import get_np_struct_type
import allo.backend.hls as hls



def test_parameterized_systolic():
    from allo.library.systolic import systolic_tile

    s = allo.customize(
        systolic_tile,
        instantiate=[int8, int8, int16, 4, 4, 4]
    )
    mod = s.build(target="vhls")
    print(mod)

if __name__ == "__main__":
    test_parameterized_systolic()
