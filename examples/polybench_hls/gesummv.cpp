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
void compute_tmp(
  int32_t v0[90],
  int32_t v1[90],
  int32_t v2[90][90],
  int32_t v3[90][90],
  int32_t v4[90],
  int32_t v5[90]
) {	// L2
  int32_t tt[90];	// L3
  for (int v7 = 0; v7 < 90; v7++) {	// L5
    tt[v7] = 0;	// L5
  }
  int32_t yy[90];	// L6
  l_load_i0: for (int i0 = 0; i0 < 90; i0++) {	// L7
    int32_t v10 = v0[i0];	// L8
    yy[i0] = v10;	// L9
  }
  l_tmp_j: for (int j = 0; j < 90; j++) {	// L11
    l_i: for (int i = 0; i < 90; i++) {	// L12
      int32_t v13 = v2[i][j];	// L13
      int32_t v14 = v4[j];	// L14
      int64_t v15 = v13;	// L15
      int64_t v16 = v14;	// L16
      int64_t v17 = v15 * v16;	// L17
      int32_t v18 = tt[i];	// L18
      int32_t v19 = v17;	// L19
      int32_t v20 = v18 + v19;	// L20
      tt[i] = v20;	// L21
      int32_t v21 = v3[i][j];	// L22
      int32_t v22 = v4[j];	// L23
      int64_t v23 = v21;	// L24
      int64_t v24 = v22;	// L25
      int64_t v25 = v23 * v24;	// L26
      int32_t v26 = yy[i];	// L27
      int32_t v27 = v25;	// L28
      int32_t v28 = v26 + v27;	// L29
      yy[i] = v28;	// L30
    }
  }
  l_store_i1: for (int i1 = 0; i1 < 90; i1++) {	// L33
    int32_t v30 = tt[i1];	// L34
    v5[i1] = v30;	// L35
    int32_t v31 = yy[i1];	// L36
    v1[i1] = v31;	// L37
  }
}

void compute_y(
  int32_t v32[90],
  int32_t v33[90],
  int32_t v34[90]
) {	// L41
  l_load_i01: for (int i01 = 0; i01 < 90; i01++) {	// L42
  #pragma HLS pipeline II=1
    int32_t v36 = v34[i01];	// L43
    int64_t v37 = v36;	// L46
    int64_t v38 = v37 * 10;	// L47
    int32_t v39 = v32[i01];	// L48
    int64_t v40 = v39;	// L49
    int64_t v41 = v40 * 10;	// L50
    ap_int<65> v42 = v38;	// L51
    ap_int<65> v43 = v41;	// L52
    ap_int<65> v44 = v42 + v43;	// L53
    int32_t v45 = v44;	// L54
    v33[i01] = v45;	// L55
  }
}

void kernel_gesummv(
  int32_t v46[90][90],
  int32_t v47[90][90],
  int32_t v48[90],
  int32_t v49[90]
) {	// L59
  int32_t y_init[90];	// L60
  for (int v51 = 0; v51 < 90; v51++) {	// L62
    y_init[v51] = 0;	// L62
  }
  int32_t y_fifo[90];	// L63
  int32_t tmp[90];	// L64
  compute_tmp(y_init, y_fifo, v46, v47, v48, tmp);	// L65
  compute_y(y_fifo, v49, tmp);	// L66
}