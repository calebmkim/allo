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
void kernel_seidel_2d(
  int32_t v0[120][120]
) {	// L2
  l_S_t_0_t: for (int t = 0; t < 40; t++) {	// L3
    l_S_i_0_i: for (int i = 0; i < 118; i++) {	// L4
      int v3 = (i + 1);	// L4
      l_S_j_0_j: for (int j = 0; j < 118; j++) {	// L5
        int v5 = (j + 1);	// L5
        int32_t v6 = v0[(v3 - 1)][(v5 - 1)];	// L6
        int32_t v7 = v0[(v3 - 1)][v5];	// L7
        ap_int<33> v8 = v6;	// L8
        ap_int<33> v9 = v7;	// L9
        ap_int<33> v10 = v8 + v9;	// L10
        int32_t v11 = v0[(v3 - 1)][(v5 + 1)];	// L11
        ap_int<34> v12 = v10;	// L12
        ap_int<34> v13 = v11;	// L13
        ap_int<34> v14 = v12 + v13;	// L14
        int32_t v15 = v0[v3][(v5 - 1)];	// L15
        ap_int<35> v16 = v14;	// L16
        ap_int<35> v17 = v15;	// L17
        ap_int<35> v18 = v16 + v17;	// L18
        int32_t v19 = v0[v3][v5];	// L19
        ap_int<36> v20 = v18;	// L20
        ap_int<36> v21 = v19;	// L21
        ap_int<36> v22 = v20 + v21;	// L22
        int32_t v23 = v0[v3][(v5 + 1)];	// L23
        ap_int<37> v24 = v22;	// L24
        ap_int<37> v25 = v23;	// L25
        ap_int<37> v26 = v24 + v25;	// L26
        int32_t v27 = v0[(v3 + 1)][(v5 - 1)];	// L27
        ap_int<38> v28 = v26;	// L28
        ap_int<38> v29 = v27;	// L29
        ap_int<38> v30 = v28 + v29;	// L30
        int32_t v31 = v0[(v3 + 1)][v5];	// L31
        ap_int<39> v32 = v30;	// L32
        ap_int<39> v33 = v31;	// L33
        ap_int<39> v34 = v32 + v33;	// L34
        int32_t v35 = v0[(v3 + 1)][(v5 + 1)];	// L35
        ap_int<40> v36 = v34;	// L36
        ap_int<40> v37 = v35;	// L37
        ap_int<40> v38 = v36 + v37;	// L38
        ap_int<40> v39 = v38 / 9;	// L41
        int32_t v40 = v39;	// L42
        v0[v3][v5] = v40;	// L43
      }
    }
  }
}
