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
void kernel_doitgen(
  int32_t v0[25][20][30],
  int32_t v1[30][30],
  int32_t v2[30]
) {	// L2
  l_S_r_q_0_r: for (int r = 0; r < 25; r++) {	// L3
    l_q: for (int q = 0; q < 20; q++) {	// L4
      l_S_p_0_p: for (int p = 0; p < 30; p++) {	// L5
        v2[p] = 0;	// L7
        l_S_s_0_s: for (int s = 0; s < 30; s++) {	// L8
          int32_t v7 = v2[p];	// L9
          int32_t v8 = v0[r][q][s];	// L10
          int32_t v9 = v1[s][p];	// L11
          int64_t v10 = v8;	// L12
          int64_t v11 = v9;	// L13
          int64_t v12 = v10 * v11;	// L14
          ap_int<65> v13 = v7;	// L15
          ap_int<65> v14 = v12;	// L16
          ap_int<65> v15 = v13 + v14;	// L17
          int32_t v16 = v15;	// L18
          v2[p] = v16;	// L19
        }
      }
      l_S_p1_2_p1: for (int p1 = 0; p1 < 30; p1++) {	// L22
        int32_t v18 = v2[p1];	// L23
        v0[r][q][p1] = v18;	// L24
      }
    }
  }
}

