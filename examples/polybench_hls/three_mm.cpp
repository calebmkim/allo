
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
  int32_t v0[40][60],
  int32_t v1[60][50],
  int32_t v2[40][50]
) {	// L4
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1 complete dim=2

  l_mm1_i0: for (int i0 = 0; i0 < 40; i0++) {	// L5
  #pragma HLS dataflow
    int32_t v4[50];	// L6
    l_j0_init: for (int j0_init = 0; j0_init < 50; j0_init++) {	// L8
    #pragma HLS pipeline II=1
      v4[j0_init] = 0;	// L9
    }
    l_S_k0_0_k0: for (int k0 = 0; k0 < 60; k0++) {	// L11
    #pragma HLS pipeline II=1
      l_j0: for (int j0 = 0; j0 < 50; j0++) {	// L12
        int32_t v8 = v0[i0][k0];	// L13
        int32_t v9 = v1[k0][j0];	// L14
        int64_t v10 = v8;	// L15
        int64_t v11 = v9;	// L16
        int64_t v12 = v10 * v11;	// L17
        int32_t v13 = v4[j0];	// L18
        int32_t v14 = v12;	// L19
        int32_t v15 = v13 + v14;	// L20
        v4[j0] = v15;	// L21
      }
    }
    l_j0_back: for (int j0_back = 0; j0_back < 50; j0_back++) {	// L24
    #pragma HLS pipeline II=1
      int32_t v17 = v4[j0_back];	// L25
      v2[i0][j0_back] = v17;	// L26
    }
  }
}

void mm2(
  int32_t v18[50][80],
  int32_t v19[80][70],
  int32_t v20[50][70]
) {	// L31
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v18 complete dim=2

  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_mm2_i1: for (int i1 = 0; i1 < 50; i1++) {	// L32
  #pragma HLS dataflow
    int32_t v22[70];	// L33
    l_j1_init: for (int j1_init = 0; j1_init < 70; j1_init++) {	// L35
    #pragma HLS pipeline II=1
      v22[j1_init] = 0;	// L36
    }
    l_S_k1_0_k1: for (int k1 = 0; k1 < 80; k1++) {	// L38
    #pragma HLS pipeline II=1
      l_j1: for (int j1 = 0; j1 < 70; j1++) {	// L39
        int32_t v26 = v18[i1][k1];	// L40
        int32_t v27 = v19[k1][j1];	// L41
        int64_t v28 = v26;	// L42
        int64_t v29 = v27;	// L43
        int64_t v30 = v28 * v29;	// L44
        int32_t v31 = v22[j1];	// L45
        int32_t v32 = v30;	// L46
        int32_t v33 = v31 + v32;	// L47
        v22[j1] = v33;	// L48
      }
    }
    l_j1_back: for (int j1_back = 0; j1_back < 70; j1_back++) {	// L51
    #pragma HLS pipeline II=1
      int32_t v35 = v22[j1_back];	// L52
      v20[i1][j1_back] = v35;	// L53
    }
  }
}

void mm3(
  int32_t v36[40][50],
  int32_t v37[50][70],
  int32_t v38[40][70]
) {	// L58
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v37 complete dim=1
  #pragma HLS array_partition variable=v37 complete dim=2

  l_mm3_i2: for (int i2 = 0; i2 < 40; i2++) {	// L59
  #pragma HLS dataflow
    int32_t v40[70];	// L60
    l_j2_init: for (int j2_init = 0; j2_init < 70; j2_init++) {	// L62
    #pragma HLS pipeline II=1
      v40[j2_init] = 0;	// L63
    }
    l_S_k2_0_k2: for (int k2 = 0; k2 < 50; k2++) {	// L65
    #pragma HLS pipeline II=1
      l_j2: for (int j2 = 0; j2 < 70; j2++) {	// L66
        int32_t v44 = v36[i2][k2];	// L67
        int32_t v45 = v37[k2][j2];	// L68
        int64_t v46 = v44;	// L69
        int64_t v47 = v45;	// L70
        int64_t v48 = v46 * v47;	// L71
        int32_t v49 = v40[j2];	// L72
        int32_t v50 = v48;	// L73
        int32_t v51 = v49 + v50;	// L74
        v40[j2] = v51;	// L75
      }
    }
    l_j2_back: for (int j2_back = 0; j2_back < 70; j2_back++) {	// L78
    #pragma HLS pipeline II=1
      int32_t v53 = v40[j2_back];	// L79
      v38[i2][j2_back] = v53;	// L80
    }
  }
}

void kernel_3mm(
  int32_t v54[40][60],
  int32_t v55[60][50],
  int32_t v56[50][80],
  int32_t v57[80][70],
  int32_t v58[40][70]
) {	// L85
  #pragma HLS array_partition variable=v55 complete dim=2

  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t out_AB[40][50];	// L86
  int32_t out_CD[50][70];	// L87
  #pragma HLS array_partition variable=out_CD complete dim=1
  #pragma HLS array_partition variable=out_CD complete dim=2

  mm1(v54, v55, out_AB);	// L89
  mm2(v56, v57, out_CD);	// L90
  mm3(out_AB, out_CD, v58);	// L91
}