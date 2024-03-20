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
void stageS(
  int32_t v0[124][116],
  int32_t v1[124],
  int32_t v2[116]
) {	// L5
  #pragma HLS array_partition variable=v0 complete dim=2

  #pragma HLS array_partition variable=v2 complete dim=1

  l_S_i0_0_i0: for (int i0 = 0; i0 < 124; i0++) {	// L6
  #pragma HLS pipeline II=1
    int32_t v4 = v1[i0];	// L7
    int32_t r;	// L8
    r = v4;	// L9
    l_S_j0_0_j0: for (int j0 = 0; j0 < 116; j0++) {	// L10
      int32_t v7 = v0[i0][j0];	// L11
      int32_t v8 = r;	// L12
      int64_t v9 = v8;	// L13
      int64_t v10 = v7;	// L14
      int64_t v11 = v9 * v10;	// L15
      int32_t v12 = v2[j0];	// L16
      int32_t v13 = v11;	// L17
      int32_t v14 = v12 + v13;	// L18
      v2[j0] = v14;	// L19
    }
  }
}

void stageQ(
  int32_t v15[124][116],
  int32_t v16[116],
  int32_t v17[124]
) {	// L24
  #pragma HLS array_partition variable=v15 complete dim=1

  #pragma HLS array_partition variable=v17 complete dim=1

  l_S_i1_0_j1: for (int j1 = 0; j1 < 116; j1++) {	// L25
  #pragma HLS pipeline II=1
    l_i1: for (int i1 = 0; i1 < 124; i1++) {	// L26
      int32_t v20 = v15[i1][j1];	// L27
      int32_t v21 = v16[j1];	// L28
      int64_t v22 = v20;	// L29
      int64_t v23 = v21;	// L30
      int64_t v24 = v22 * v23;	// L31
      int32_t v25 = v17[i1];	// L32
      int32_t v26 = v24;	// L33
      int32_t v27 = v25 + v26;	// L34
      v17[i1] = v27;	// L35
    }
  }
}

void kernel_bicg(
  int32_t v28[124][116],
  int32_t v29[124][116],
  int32_t v30[116],
  int32_t v31[124],
  int32_t v32[124],
  int32_t v33[116]
) {	// L40
  #pragma HLS array_partition variable=v28 complete dim=2

  #pragma HLS array_partition variable=v29 complete dim=1

  #pragma HLS array_partition variable=v32 complete dim=1

  #pragma HLS array_partition variable=v33 complete dim=1

  stageS(v28, v31, v33);	// L41
  stageQ(v29, v30, v32);	// L42
}