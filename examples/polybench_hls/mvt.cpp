
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
void stageA(
  int32_t v0[120],
  int32_t v1[120],
  int32_t v2[120][120],
  int32_t v3[120]
) {	// L5
  #pragma HLS array_partition variable=v2 complete dim=2

  #pragma HLS array_partition variable=v3 complete dim=1

  l_A_i0: for (int i0 = 0; i0 < 120; i0++) {	// L6
  #pragma HLS pipeline II=1
    int32_t v5 = v0[i0];	// L7
    int32_t x;	// L8
    x = v5;	// L9
    l_S_j0_0_j0: for (int j0 = 0; j0 < 120; j0++) {	// L10
      int32_t v8 = v2[i0][j0];	// L11
      int32_t v9 = v3[j0];	// L12
      int64_t v10 = v8;	// L13
      int64_t v11 = v9;	// L14
      int64_t v12 = v10 * v11;	// L15
      int32_t v13 = v12;	// L16
      int32_t v14 = x;	// L17
      int32_t v15 = v14 + v13;	// L18
      x = v15;	// L19
    }
    int32_t v16 = x;	// L21
    v1[i0] = v16;	// L22
  }
}

void stageB(
  int32_t v17[120],
  int32_t v18[120],
  int32_t v19[120][120],
  int32_t v20[120]
) {	// L26
  #pragma HLS array_partition variable=v19 complete dim=1

  #pragma HLS array_partition variable=v20 complete dim=1

  l_B_i1: for (int i1 = 0; i1 < 120; i1++) {	// L27
  #pragma HLS pipeline II=1
    int32_t v22 = v17[i1];	// L28
    int32_t x1;	// L29
    x1 = v22;	// L30
    l_S_j1_0_j1: for (int j1 = 0; j1 < 120; j1++) {	// L31
      int32_t v25 = v19[j1][i1];	// L32
      int32_t v26 = v20[j1];	// L33
      int64_t v27 = v25;	// L34
      int64_t v28 = v26;	// L35
      int64_t v29 = v27 * v28;	// L36
      int32_t v30 = v29;	// L37
      int32_t v31 = x1;	// L38
      int32_t v32 = v31 + v30;	// L39
      x1 = v32;	// L40
    }
    int32_t v33 = x1;	// L42
    v18[i1] = v33;	// L43
  }
}

void kernel_mvt(
  int32_t v34[120][120],
  int32_t v35[120][120],
  int32_t v36[120],
  int32_t v37[120],
  int32_t v38[120],
  int32_t v39[120],
  int32_t v40[120],
  int32_t v41[120]
) {	// L47
  #pragma HLS array_partition variable=v34 complete dim=2

  #pragma HLS array_partition variable=v35 complete dim=1

  #pragma HLS array_partition variable=v36 complete dim=1

  #pragma HLS array_partition variable=v37 complete dim=1

  stageA(v38, v40, v34, v36);	// L48
  stageB(v39, v41, v35, v37);	// L49
}
