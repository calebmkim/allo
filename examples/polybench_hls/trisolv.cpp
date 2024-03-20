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
void kernel_trisolv(
  int32_t v0[120][120],
  int32_t v1[120],
  int32_t v2[120]
) {	// L3
  l_S_i_0_i: for (int i = 0; i < 120; i++) {	// L4
    int32_t v4 = v1[i];	// L5
    v2[i] = v4;	// L6
    l_S_j_0_j: for (int j = 0; j < i; j++) {	// L7
      int32_t v6 = v0[i][j];	// L8
      int32_t v7 = v2[j];	// L9
      int64_t v8 = v6;	// L10
      int64_t v9 = v7;	// L11
      int64_t v10 = v8 * v9;	// L12
      int32_t v11 = v2[i];	// L13
      int32_t v12 = v10;	// L14
      int32_t v13 = v11 - v12;	// L15
      v2[i] = v13;	// L16
    }
    int32_t v14 = v2[i];	// L18
    int32_t v15 = v0[i][i];	// L19
    int32_t v16 = v14 / v15;	// L20
    v2[i] = v16;	// L21
  }
}