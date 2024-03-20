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
void mm1(
  int32_t v0[60][80],
  int32_t v1[80][70],
  int32_t v2[60][70]
) {	// L2
  l_mm1_i0: for (int i0 = 0; i0 < 60; i0++) {	// L3
    int32_t v4[70];	// L4
    l_j0_init: for (int j0_init = 0; j0_init < 70; j0_init++) {	// L6
    #pragma HLS pipeline II=1
      v4[j0_init] = 0;	// L7
    }
    l_S_k0_0_k0: for (int k0 = 0; k0 < 80; k0++) {	// L9
    #pragma HLS pipeline II=1
      l_j0: for (int j0 = 0; j0 < 70; j0++) {	// L10
        int32_t v8 = v0[i0][k0];	// L11
        int32_t v9 = v1[k0][j0];	// L12
        int64_t v10 = v8;	// L13
        int64_t v11 = v9;	// L14
        int64_t v12 = v10 * v11;	// L15
        int32_t v13 = v4[j0];	// L16
        int32_t v14 = v12;	// L17
        int32_t v15 = v13 + v14;	// L18
        v4[j0] = v15;	// L19
      }
    }
    l_j0_back: for (int j0_back = 0; j0_back < 70; j0_back++) {	// L22
    #pragma HLS pipeline II=1
      int32_t v17 = v4[j0_back];	// L23
      v2[i0][j0_back] = v17;	// L24
    }
  }
}

void ele_add(
  int32_t v18[60][70],
  int32_t v19[60][70],
  int32_t v20[60][70]
) {	// L29
  l_S_i2_j2_0_i2: for (int i2 = 0; i2 < 60; i2++) {	// L30
    l_j2: for (int j2 = 0; j2 < 70; j2++) {	// L31
    #pragma HLS pipeline II=1
      int32_t v23 = v19[i2][j2];	// L32
      int64_t v24 = v23;	// L35
      int64_t v25 = v24 * 10;	// L36
      int32_t v26 = v18[i2][j2];	// L37
      ap_int<65> v27 = v25;	// L38
      ap_int<65> v28 = v26;	// L39
      ap_int<65> v29 = v27 + v28;	// L40
      int32_t v30 = v29;	// L41
      v20[i2][j2] = v30;	// L42
    }
  }
}

void kernel_gemm(
  int32_t v31[60][80],
  int32_t v32[80][70],
  int32_t v33[60][70],
  int32_t v34[60][70]
) {	// L47
  int32_t out_AB[60][70];	// L48
  mm1(v31, v32, out_AB);	// L49
  ele_add(out_AB, v33, v34);	// L50
}