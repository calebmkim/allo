
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
void stage_M(
  int32_t v0[116][124],
  int32_t v1[124],
  int32_t v2[116]
) {	// L2
  l_S_m_0_r: for (int r = 0; r < 124; r++) {	// L3
    l_m: for (int m = 0; m < 116; m++) {	// L4
    #pragma HLS pipeline II=1
      int32_t v5 = v0[m][r];	// L5
      int32_t v6 = v1[r];	// L6
      int64_t v7 = v5;	// L7
      int64_t v8 = v6;	// L8
      int64_t v9 = v7 * v8;	// L9
      int32_t v10 = v2[m];	// L10
      int32_t v11 = v9;	// L11
      int32_t v12 = v10 + v11;	// L12
      v2[m] = v12;	// L13
    }
  }
}

void stage_N(
  int32_t v13[116][124],
  int32_t v14[116],
  int32_t v15[124]
) {	// L18
  l_S_n_0_k: for (int k = 0; k < 116; k++) {	// L19
    l_n: for (int n = 0; n < 124; n++) {	// L20
    #pragma HLS pipeline II=1
      int32_t v18 = v13[k][n];	// L21
      int32_t v19 = v14[k];	// L22
      int64_t v20 = v18;	// L23
      int64_t v21 = v19;	// L24
      int64_t v22 = v20 * v21;	// L25
      int32_t v23 = v15[n];	// L26
      int32_t v24 = v22;	// L27
      int32_t v25 = v23 + v24;	// L28
      v15[n] = v25;	// L29
    }
  }
}

void kernel_atax(
  int32_t v26[116][124],
  int32_t v27[124],
  int32_t v28[124]
) {	// L34
  int32_t out_Ax[116];	// L35
  for (int v30 = 0; v30 < 116; v30++) {	// L37
    out_Ax[v30] = 0;	// L37
  }
  stage_M(v26, v27, out_Ax);	// L38
  stage_N(v26, out_Ax, v28);	// L39
}

