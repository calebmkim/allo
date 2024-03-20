
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
  int32_t v12[80][60],
  int32_t v13[80][60],
  int32_t v14[80][80],
  int32_t v15[80][80]
) {	// L24
  #pragma HLS array_partition variable=v10 complete dim=1

  #pragma HLS array_partition variable=v13 complete dim=1

  #pragma HLS array_partition variable=v14 complete dim=2

  #pragma HLS array_partition variable=v15 complete dim=2

  int32_t buffer[80][80];	// L25
  #pragma HLS array_partition variable=buffer complete dim=2

  l_load_i01: for (int i01 = 0; i01 < 80; i01++) {	// L26
  #pragma HLS pipeline II=1
    l_j01: for (int j01 = 0; j01 < 80; j01++) {	// L27
      int32_t v19 = v14[i01][j01];	// L28
      buffer[i01][j01] = v19;	// L29
    }
  }
  l_sum_i1: for (int i1 = 0; i1 < 80; i1++) {	// L32
    l_k1: for (int k1 = 0; k1 < 60; k1++) {	// L33
    #pragma HLS pipeline II=1
      l_j1: for (int j1 = 0; j1 < 80; j1++) {	// L34
        bool v23 = j1 <= i1;	// L35
        if (v23) {	// L36
          int32_t v24 = v10[j1][k1];	// L37
          int64_t v25 = v24;	// L38
          int64_t v26 = v25 * 15;	// L41
          int32_t v27 = v12[i1][k1];	// L42
          ap_int<96> v28 = v26;	// L43
          ap_int<96> v29 = v27;	// L44
          ap_int<96> v30 = v28 * v29;	// L45
          int32_t v31 = v13[j1][k1];	// L46
          int64_t v32 = v31;	// L47
          int64_t v33 = v32 * 15;	// L48
          int32_t v34 = v11[i1][k1];	// L49
          ap_int<96> v35 = v33;	// L50
          ap_int<96> v36 = v34;	// L51
          ap_int<96> v37 = v35 * v36;	// L52
          ap_int<97> v38 = v30;	// L53
          ap_int<97> v39 = v37;	// L54
          ap_int<97> v40 = v38 + v39;	// L55
          int32_t v41 = buffer[i1][j1];	// L56
          int32_t v42 = v40;	// L57
          int32_t v43 = v41 + v42;	// L58
          buffer[i1][j1] = v43;	// L59
        }
      }
    }
  }
  l_store_i2: for (int i2 = 0; i2 < 80; i2++) {	// L64
  #pragma HLS pipeline II=1
    l_j2: for (int j2 = 0; j2 < 80; j2++) {	// L65
      int32_t v46 = buffer[i2][j2];	// L66
      v15[i2][j2] = v46;	// L67
    }
  }
}

void kernel_syr2k(
  int32_t v47[80][60],
  int32_t v48[80][60],
  int32_t v49[80][60],
  int32_t v50[80][60],
  int32_t v51[80][80],
  int32_t v52[80][80]
) {	// L72
  #pragma HLS array_partition variable=v47 complete dim=1

  #pragma HLS array_partition variable=v50 complete dim=1

  #pragma HLS array_partition variable=v51 complete dim=2

  #pragma HLS array_partition variable=v52 complete dim=2

  int32_t C[80][80];	// L73
  #pragma HLS array_partition variable=C complete dim=2

  update_C(v51, C);	// L74
  compute_sum(v47, v48, v49, v50, C, v52);	// L75
}