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
void kernel_gemver(
  int32_t v0[40][40],
  int32_t v1[40],
  int32_t v2[40],
  int32_t v3[40],
  int32_t v4[40],
  int32_t v5[40],
  int32_t v6[40],
  int32_t v7[40],
  int32_t v8[40]
) {	// L2
  l_S_i_j_0_i: for (int i = 0; i < 40; i++) {	// L3
    l_j: for (int j = 0; j < 40; j++) {	// L4
      int32_t v11 = v0[i][j];	// L5
      int32_t v12 = v1[i];	// L6
      int32_t v13 = v3[j];	// L7
      int64_t v14 = v12;	// L8
      int64_t v15 = v13;	// L9
      int64_t v16 = v14 * v15;	// L10
      ap_int<65> v17 = v11;	// L11
      ap_int<65> v18 = v16;	// L12
      ap_int<65> v19 = v17 + v18;	// L13
      int32_t v20 = v2[i];	// L14
      int32_t v21 = v4[j];	// L15
      int64_t v22 = v20;	// L16
      int64_t v23 = v21;	// L17
      int64_t v24 = v22 * v23;	// L18
      ap_int<66> v25 = v19;	// L19
      ap_int<66> v26 = v24;	// L20
      ap_int<66> v27 = v25 + v26;	// L21
      int32_t v28 = v27;	// L22
      v0[i][j] = v28;	// L23
    }
  }
  l_S_i_j_1_i1: for (int i1 = 0; i1 < 40; i1++) {	// L26
    l_j1: for (int j1 = 0; j1 < 40; j1++) {	// L27
      int32_t v31 = v5[i1];	// L28
      int32_t v32 = v0[j1][i1];	// L29
      int64_t v33 = v32;	// L32
      int64_t v34 = v33 * 10;	// L33
      int32_t v35 = v6[j1];	// L34
      ap_int<96> v36 = v34;	// L35
      ap_int<96> v37 = v35;	// L36
      ap_int<96> v38 = v36 * v37;	// L37
      ap_int<97> v39 = v31;	// L38
      ap_int<97> v40 = v38;	// L39
      ap_int<97> v41 = v39 + v40;	// L40
      int32_t v42 = v41;	// L41
      v5[i1] = v42;	// L42
    }
  }
  l_S_i_2_i2: for (int i2 = 0; i2 < 40; i2++) {	// L45
    int32_t v44 = v5[i2];	// L46
    int32_t v45 = v8[i2];	// L47
    ap_int<33> v46 = v44;	// L48
    ap_int<33> v47 = v45;	// L49
    ap_int<33> v48 = v46 + v47;	// L50
    int32_t v49 = v48;	// L51
    v5[i2] = v49;	// L52
  }
  l_S_i_j_3_i3: for (int i3 = 0; i3 < 40; i3++) {	// L54
    l_j2: for (int j2 = 0; j2 < 40; j2++) {	// L55
      int32_t v52 = v7[i3];	// L56
      int32_t v53 = v0[i3][j2];	// L57
      int64_t v54 = v53;	// L60
      int64_t v55 = v54 * 10;	// L61
      int32_t v56 = v5[j2];	// L62
      ap_int<96> v57 = v55;	// L63
      ap_int<96> v58 = v56;	// L64
      ap_int<96> v59 = v57 * v58;	// L65
      ap_int<97> v60 = v52;	// L66
      ap_int<97> v61 = v59;	// L67
      ap_int<97> v62 = v60 + v61;	// L68
      int32_t v63 = v62;	// L69
      v7[i3] = v63;	// L70
    }
  }
}