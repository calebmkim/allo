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
void kernel_floyd_warshall(
  int32_t v0[180][180]
) {	// L2
  l_S_k_i_j_0_k: for (int k = 0; k < 180; k++) {	// L3
    l_i: for (int i = 0; i < 180; i++) {	// L4
      l_j: for (int j = 0; j < 180; j++) {	// L5
        int32_t v4 = v0[i][k];	// L6
        int32_t v5 = v0[k][j];	// L7
        ap_int<33> v6 = v4;	// L8
        ap_int<33> v7 = v5;	// L9
        ap_int<33> v8 = v6 + v7;	// L10
        int32_t v9 = v8;	// L11
        int32_t path_;	// L12
        path_ = v9;	// L13
        int32_t v11 = v0[i][j];	// L14
        int32_t v12 = path_;	// L15
        bool v13 = v11 >= v12;	// L16
        if (v13) {	// L17
          int32_t v14 = path_;	// L18
          v0[i][j] = v14;	// L19
        }
      }
    }
  }
}