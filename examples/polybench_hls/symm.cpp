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
void compute_sum(
  int32_t v0[60][60],
  int32_t v1[60][80],
  int32_t v2[60][80]
) {	// L5
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=24

  #pragma HLS array_partition variable=v2 complete dim=2

  l_sum_i1: for (int i1 = 0; i1 < 60; i1++) {	// L6
    int32_t v4[80];	// L7
    l_j1_init: for (int j1_init = 0; j1_init < 80; j1_init++) {	// L9
    #pragma HLS pipeline II=1
      v4[j1_init] = 0;	// L10
    }
    l_k1_k1: for (int k1 = 0; k1 < 60; k1++) {	// L12
      l_j1: for (int j1 = 0; j1 < 80; j1++) {	// L13
      #pragma HLS pipeline II=1
      #pragma HLS unroll factor=24
        bool v8 = k1 < i1;	// L14
        if (v8) {	// L15
          int32_t v9 = v1[k1][j1];	// L16
          int32_t v10 = v0[i1][k1];	// L17
          int64_t v11 = v9;	// L18
          int64_t v12 = v10;	// L19
          int64_t v13 = v11 * v12;	// L20
          int32_t v14 = v4[j1];	// L21
          int32_t v15 = v13;	// L22
          int32_t v16 = v14 + v15;	// L23
          v4[j1] = v16;	// L24
        }
      }
    }
    l_j1_back: for (int j1_back = 0; j1_back < 80; j1_back++) {	// L28
    #pragma HLS pipeline II=1
      int32_t v18 = v4[j1_back];	// L29
      v2[i1][j1_back] = v18;	// L30
    }
  }
}

void update_C(
  int32_t v19[60][60],
  int32_t v20[60][80],
  int32_t v21[60][80],
  int32_t v22[60][80]
) {	// L35
  #pragma HLS array_partition variable=v20 complete dim=2

  #pragma HLS array_partition variable=v21 complete dim=2

  #pragma HLS array_partition variable=v22 complete dim=2

  l_S_i_0_i: for (int i = 0; i < 60; i++) {	// L36
    l_S_k_0_k: for (int k = 0; k < i; k++) {	// L37
    #pragma HLS pipeline II=1
      l_S_j_0_j: for (int j = 0; j < 80; j++) {	// L38
        int32_t v26 = v22[k][j];	// L39
        int32_t v27 = v20[i][j];	// L40
        int64_t v28 = v27;	// L43
        int64_t v29 = v28 * 15;	// L44
        int32_t v30 = v19[i][k];	// L45
        ap_int<96> v31 = v29;	// L46
        ap_int<96> v32 = v30;	// L47
        ap_int<96> v33 = v31 * v32;	// L48
        ap_int<97> v34 = v26;	// L49
        ap_int<97> v35 = v33;	// L50
        ap_int<97> v36 = v34 + v35;	// L51
        int32_t v37 = v36;	// L52
        v22[k][j] = v37;	// L53
      }
    }
    l_S_j1_2_j11: for (int j11 = 0; j11 < 80; j11++) {	// L56
    #pragma HLS pipeline II=1
      int32_t v39 = v22[i][j11];	// L57
      int64_t v40 = v39;	// L60
      int64_t v41 = v40 * 12;	// L61
      int32_t v42 = v20[i][j11];	// L62
      int64_t v43 = v42;	// L65
      int64_t v44 = v43 * 15;	// L66
      int32_t v45 = v19[i][i];	// L67
      ap_int<96> v46 = v44;	// L68
      ap_int<96> v47 = v45;	// L69
      ap_int<96> v48 = v46 * v47;	// L70
      ap_int<97> v49 = v41;	// L71
      ap_int<97> v50 = v48;	// L72
      ap_int<97> v51 = v49 + v50;	// L73
      int32_t v52 = v21[i][j11];	// L74
      int64_t v53 = v52;	// L75
      int64_t v54 = v53 * 15;	// L76
      ap_int<98> v55 = v51;	// L77
      ap_int<98> v56 = v54;	// L78
      ap_int<98> v57 = v55 + v56;	// L79
      int32_t v58 = v57;	// L80
      v22[i][j11] = v58;	// L81
    }
  }
}

void kernel_symm(
  int32_t v59[60][60],
  int32_t v60[60][60],
  int32_t v61[60][80],
  int32_t v62[60][80],
  int32_t v63[60][80]
) {	// L86
  #pragma HLS array_partition variable=v61 cyclic dim=2 factor=24

  #pragma HLS array_partition variable=v62 complete dim=2

  #pragma HLS array_partition variable=v63 complete dim=2

  int32_t summ[60][80];	// L87
  #pragma HLS array_partition variable=summ complete dim=2

  compute_sum(v59, v61, summ);	// L88
  update_C(v60, v62, summ, v63);	// L89
}
