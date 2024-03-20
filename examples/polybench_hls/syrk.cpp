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
void update_C(
  int32_t v0[80][80],
  int32_t v1[80][80]
) {	// L4
  #pragma HLS array_partition variable=v0 complete dim=2

  #pragma HLS array_partition variable=v1 complete dim=2

  l_update_i0: for (int i0 = 0; i0 < 80; i0++) {	// L5
  #pragma HLS pipeline II=1
    l_j0: for (int j0 = 0; j0 < 80; j0++) {	// L6
      bool v4 = j0 <= i0;	// L7
      if (v4) {	// L8
        int32_t v5 = v0[i0][j0];	// L9
        int64_t v6 = v5;	// L12
        int64_t v7 = v6 * 12;	// L13
        int32_t v8 = v7;	// L14
        v1[i0][j0] = v8;	// L15
      } else {
        int32_t v9 = v0[i0][j0];	// L17
        v1[i0][j0] = v9;	// L18
      }
    }
  }
}

void compute_sum(
  int32_t v10[80][60],
  int32_t v11[80][60],
  int32_t v12[80][80],
  int32_t v13[80][80]
) {	// L24
  #pragma HLS array_partition variable=v11 complete dim=1

  #pragma HLS array_partition variable=v12 complete dim=2

  #pragma HLS array_partition variable=v13 complete dim=2

  int32_t buffer[80][80];	// L25
  #pragma HLS array_partition variable=buffer complete dim=2

  l_load_i01: for (int i01 = 0; i01 < 80; i01++) {	// L26
  #pragma HLS pipeline II=1
    l_j01: for (int j01 = 0; j01 < 80; j01++) {	// L27
      int32_t v17 = v12[i01][j01];	// L28
      buffer[i01][j01] = v17;	// L29
    }
  }
  l_sum_i1: for (int i1 = 0; i1 < 80; i1++) {	// L32
    l_k1: for (int k1 = 0; k1 < 60; k1++) {	// L33
    #pragma HLS pipeline II=1
      l_j1: for (int j1 = 0; j1 < 80; j1++) {	// L34
        bool v21 = j1 <= i1;	// L35
        if (v21) {	// L36
          int32_t v22 = v10[i1][k1];	// L37
          int64_t v23 = v22;	// L40
          int64_t v24 = v23 * 15;	// L41
          int32_t v25 = v11[j1][k1];	// L42
          ap_int<96> v26 = v24;	// L43
          ap_int<96> v27 = v25;	// L44
          ap_int<96> v28 = v26 * v27;	// L45
          int32_t v29 = buffer[i1][j1];	// L46
          int32_t v30 = v28;	// L47
          int32_t v31 = v29 + v30;	// L48
          buffer[i1][j1] = v31;	// L49
        }
      }
    }
  }
  l_store_i2: for (int i2 = 0; i2 < 80; i2++) {	// L54
  #pragma HLS pipeline II=1
    l_j2: for (int j2 = 0; j2 < 80; j2++) {	// L55
      int32_t v34 = buffer[i2][j2];	// L56
      v13[i2][j2] = v34;	// L57
    }
  }
}

void kernel_syr2k(
  int32_t v35[80][60],
  int32_t v36[80][60],
  int32_t v37[80][80],
  int32_t v38[80][80]
) {	// L62
  #pragma HLS array_partition variable=v36 complete dim=1

  #pragma HLS array_partition variable=v37 complete dim=2

  #pragma HLS array_partition variable=v38 complete dim=2

  int32_t C[80][80];	// L63
  #pragma HLS array_partition variable=C complete dim=2

  update_C(v37, C);	// L64
  compute_sum(v35, v36, C, v38);	// L65
}