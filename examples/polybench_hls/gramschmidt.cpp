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
void kernel_gramschmidt(
  int32_t v0[60][80],
  int32_t v1[60][80],
  int32_t v2[80][80]
) {	// L3
  l_S_k_0_k: for (int k = 0; k < 80; k++) {	// L4
    int32_t nrm;	// L5
    nrm = 0;	// L7
    l_S_i_0_i: for (int i = 0; i < 60; i++) {	// L8
      int32_t v6 = v0[i][k];	// L9
      int64_t v7 = v6;	// L11
      int64_t v8 = v7 * v7;	// L13
      int32_t v9 = v8;	// L14
      int32_t v10 = nrm;	// L15
      int32_t v11 = v10 + v9;	// L16
      nrm = v11;	// L17
    }
    int32_t v12 = nrm;	// L19
    v2[k][k] = v12;	// L20
    l_S_i_1_i1: for (int i1 = 0; i1 < 60; i1++) {	// L21
      int32_t v14 = v0[i1][k];	// L22
      int32_t v15 = v2[k][k];	// L23
      int32_t v16 = v14 / v15;	// L24
      v1[i1][k] = v16;	// L25
    }
    l_S_j_2_j: for (int j = 0; j < ((-k) + 79); j++) {	// L27
      int v18 = ((k + j) + 1);	// L27
      v2[k][v18] = 0;	// L30
      l_S_i_2_i2: for (int i2 = 0; i2 < 60; i2++) {	// L31
        int32_t v20 = v1[i2][k];	// L32
        int32_t v21 = v0[i2][v18];	// L33
        int64_t v22 = v20;	// L34
        int64_t v23 = v21;	// L35
        int64_t v24 = v22 * v23;	// L36
        int32_t v25 = v2[k][v18];	// L37
        int32_t v26 = v24;	// L38
        int32_t v27 = v25 + v26;	// L39
        v2[k][v18] = v27;	// L40
      }
      l_S_i_3_i3: for (int i3 = 0; i3 < 60; i3++) {	// L42
        int32_t v29 = v1[i3][k];	// L43
        int32_t v30 = v2[k][v18];	// L44
        int64_t v31 = v29;	// L45
        int64_t v32 = v30;	// L46
        int64_t v33 = v31 * v32;	// L47
        int32_t v34 = v0[i3][v18];	// L48
        int32_t v35 = v33;	// L49
        int32_t v36 = v34 - v35;	// L50
        v0[i3][v18] = v36;	// L51
      }
    }
  }
}