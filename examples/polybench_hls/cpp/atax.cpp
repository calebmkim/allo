
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
void stage_M(
  ap_fixed<32, 16> v0[116][124],
  ap_fixed<32, 16> v1[124],
  ap_fixed<32, 16> v2[116]
) {	// L2
  l_S_m_0_r: for (int r = 0; r < 124; r++) {	// L3
    l_m: for (int m = 0; m < 116; m++) {	// L4
    #pragma HLS pipeline II=1
      ap_fixed<32, 16> v5 = v0[m][r];	// L5
      ap_fixed<32, 16> v6 = v1[r];	// L6
      ap_fixed<64, 32> v7 = v5;	// L7
      ap_fixed<64, 32> v8 = v6;	// L8
      ap_fixed<64, 32> v9 = v7 * v8;	// L9
      ap_fixed<32, 16> v10 = v2[m];	// L10
      ap_fixed<32, 16> v11 = v9;	// L11
      ap_fixed<32, 16> v12 = v10 + v11;	// L12
      v2[m] = v12;	// L13
    }
  }
}

void stage_N(
  ap_fixed<32, 16> v13[116][124],
  ap_fixed<32, 16> v14[116],
  ap_fixed<32, 16> v15[124]
) {	// L18
  l_S_n_0_k: for (int k = 0; k < 116; k++) {	// L19
    l_n: for (int n = 0; n < 124; n++) {	// L20
    #pragma HLS pipeline II=1
      ap_fixed<32, 16> v18 = v13[k][n];	// L21
      ap_fixed<32, 16> v19 = v14[k];	// L22
      ap_fixed<64, 32> v20 = v18;	// L23
      ap_fixed<64, 32> v21 = v19;	// L24
      ap_fixed<64, 32> v22 = v20 * v21;	// L25
      ap_fixed<32, 16> v23 = v15[n];	// L26
      ap_fixed<32, 16> v24 = v22;	// L27
      ap_fixed<32, 16> v25 = v23 + v24;	// L28
      v15[n] = v25;	// L29
    }
  }
}

void kernel_atax(
  ap_fixed<32, 16> v26[116][124],
  ap_fixed<32, 16> v27[124],
  ap_fixed<32, 16> v28[124]
) {	// L34
  ap_fixed<32, 16> v29 = 0.000000;	// L36
  ap_fixed<32, 16> out_Ax[116];	// L37
  for (int v31 = 0; v31 < 116; v31++) {	// L38
    out_Ax[v31] = v29;	// L38
  }
  stage_M(v26, v27, out_Ax);	// L39
  stage_N(v26, out_Ax, v28);	// L40
}
