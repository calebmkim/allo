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
  int32_t v0[40][70],
  int32_t v1[70][50],
  int32_t v2[40][50]
) {	// L2
  #pragma HLS dataflow
  l_mm1_i0: for (int i0 = 0; i0 < 40; i0++) {	// L3
  #pragma HLS dataflow
    int32_t v4[50];	// L4
    l_j0_init: for (int j0_init = 0; j0_init < 50; j0_init++) {	// L6
    #pragma HLS pipeline II=1
      v4[j0_init] = 0;	// L7
    }
    l_S_k0_0_k0: for (int k0 = 0; k0 < 70; k0++) {	// L9
    #pragma HLS pipeline II=1
      l_j0: for (int j0 = 0; j0 < 50; j0++) {	// L10
        int32_t v8 = v0[i0][k0];	// L11
        int32_t v9 = v1[k0][j0];	// L12
        int64_t v10 = v8;	// L13
        int64_t v11 = v9;	// L14
        int64_t v12 = v10 * v11;	// L15
        int32_t v13 = v4[j0];	// L16
        int32_t v14 = v12;	// L17
        int32_t v15 = v13 + v14;	// L18
        v4[j0] = v15;	// L19
      }
    }
    l_j0_back: for (int j0_back = 0; j0_back < 50; j0_back++) {	// L22
    #pragma HLS pipeline II=1
      int32_t v17 = v4[j0_back];	// L23
      v2[i0][j0_back] = v17;	// L24
    }
  }
}

void mm2(
  int32_t v18[40][50],
  int32_t v19[50][80],
  int32_t v20[40][80]
) {	// L29
  #pragma HLS dataflow
  l_mm2_i1: for (int i1 = 0; i1 < 40; i1++) {	// L30
  #pragma HLS dataflow
    int32_t v22[80];	// L31
    l_j1_init: for (int j1_init = 0; j1_init < 80; j1_init++) {	// L33
    #pragma HLS pipeline II=1
      v22[j1_init] = 0;	// L34
    }
    l_S_k1_0_k1: for (int k1 = 0; k1 < 50; k1++) {	// L36
    #pragma HLS pipeline II=1
      l_j1: for (int j1 = 0; j1 < 80; j1++) {	// L37
        int32_t v26 = v18[i1][k1];	// L38
        int32_t v27 = v19[k1][j1];	// L39
        int64_t v28 = v26;	// L40
        int64_t v29 = v27;	// L41
        int64_t v30 = v28 * v29;	// L42
        int32_t v31 = v22[j1];	// L43
        int32_t v32 = v30;	// L44
        int32_t v33 = v31 + v32;	// L45
        v22[j1] = v33;	// L46
      }
    }
    l_j1_back: for (int j1_back = 0; j1_back < 80; j1_back++) {	// L49
    #pragma HLS pipeline II=1
      int32_t v35 = v22[j1_back];	// L50
      v20[i1][j1_back] = v35;	// L51
    }
  }
}

void ele_add(
  int32_t v36[40][80],
  int32_t v37[40][80],
  int32_t v38[40][80]
) {	// L56
  l_S_i2_j2_0_i2: for (int i2 = 0; i2 < 40; i2++) {	// L57
    l_j2: for (int j2 = 0; j2 < 80; j2++) {	// L58
    #pragma HLS pipeline II=1
      int32_t v41 = v36[i2][j2];	// L59
      int64_t v42 = v41;	// L60
      int64_t v43 = v42 * 5;	// L63
      int32_t v44 = v37[i2][j2];	// L64
      ap_int<65> v45 = v43;	// L69
      ap_int<65> v46 = v44;	// L70
      ap_int<65> v47 = v45 + v46;	// L71
      int32_t v48 = v47;	// L72
      v38[i2][j2] = v48;	// L73
    }
  }
}

void kernel_2mm(
  int32_t v49[40][70],
  int32_t v50[70][50],
  int32_t v51[50][80],
  int32_t v52[40][80],
  int32_t v53[40][80]
) {	// L78
  int32_t out_AB[40][50];	// L79
  int32_t out_ABC[40][80];	// L80
  mm1(v49, v50, out_AB);	// L82
  mm2(out_AB, v51, out_ABC);	// L83
  ele_add(out_ABC, v52, v53);	// L84
}