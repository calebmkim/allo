//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
void S0(
  int32_t v0[60][60],
  int32_t v1[60][80]
) {	// L3
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=20

  l_update_i1: for (int i1 = 0; i1 < 60; i1++) {	// L4
    l_S_k1_0_k1: for (int k1 = 0; k1 < 60; k1++) {	// L5
      l_j1: for (int j1 = 0; j1 < 80; j1++) {	// L6
      #pragma HLS pipeline II=1
      #pragma HLS unroll factor=20
        bool v5 = k1 > i1;	// L7
        if (v5) {	// L8
          int32_t v6 = v0[k1][i1];	// L9
          int32_t v7 = v1[k1][j1];	// L10
          int64_t v8 = v6;	// L11
          int64_t v9 = v7;	// L12
          int64_t v10 = v8 * v9;	// L13
          int32_t v11 = v1[i1][j1];	// L14
          int32_t v12 = v10;	// L15
          int32_t v13 = v11 + v12;	// L16
          v1[i1][j1] = v13;	// L17
        }
      }
    }
  }
}

void S1(
  int32_t v14[60][80]
) {	// L24
  #pragma HLS array_partition variable=v14 cyclic dim=2 factor=20

  l_mul_i0: for (int i0 = 0; i0 < 60; i0++) {	// L25
    l_j0: for (int j0 = 0; j0 < 80; j0++) {	// L26
    #pragma HLS pipeline II=1
    #pragma HLS unroll factor=20
      int32_t v17 = v14[i0][j0];	// L27
      int64_t v18 = v17;	// L28
      int64_t v19 = v18 * 15;	// L31
      int32_t v20 = v19;	// L32
      v14[i0][j0] = v20;	// L33
    }
  }
}

void kernel_trmm(
  int32_t v21[60][60],
  int32_t v22[60][80]
) {	// L38
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=20

  S0(v21, v22);	// L39
  S1(v22);	// L40
}
