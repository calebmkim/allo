
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
void kernel_nussinov(
  int32_t v0[180],
  int32_t v1[180][180]
) {	// L2
  l_S_i_inv_0_i_inv: for (int i_inv = 0; i_inv < 180; i_inv++) {	// L3
    ap_int<34> v3 = i_inv;	// L10
    ap_int<34> v4 = 179 - v3;	// L11
    int v5 = v4;	// L12
    int i;	// L13
    i = v5;	// L14
    int v7 = i;	// L15
    ap_int<34> v8 = v7;	// L16
    ap_int<34> v9 = v8 + 1;	// L19
    int v10 = v9;	// L20
    for (int v11 = v10; v11 < 180; v11 += 1) {	// L25
      ap_int<34> v12 = v11;	// L26
      ap_int<34> v13 = v12 - 1;	// L29
      bool v14 = v13 >= 0;	// L32
      if (v14) {	// L33
        int v15 = i;	// L34
        int32_t v16 = v1[v15][v11];	// L35
        int v17 = v13;	// L41
        int32_t v18 = v1[v15][v17];	// L42
        bool v19 = v16 < v18;	// L43
        if (v19) {	// L44
          int v20 = i;	// L45
          int32_t v21 = v1[v20][v17];	// L51
          v1[v20][v11] = v21;	// L53
        }
      }
      int v22 = i;	// L56
      ap_int<34> v23 = v22;	// L57
      ap_int<34> v24 = v23 + 1;	// L60
      bool v25 = v24 < 180;	// L63
      if (v25) {	// L64
        int v26 = i;	// L65
        int32_t v27 = v1[v26][v11];	// L66
        ap_int<34> v28 = v26;	// L68
        ap_int<34> v29 = v28 + 1;	// L71
        int v30 = v29;	// L72
        int32_t v31 = v1[v30][v11];	// L73
        bool v32 = v27 < v31;	// L74
        if (v32) {	// L75
          int v33 = i;	// L76
          ap_int<34> v34 = v33;	// L77
          ap_int<34> v35 = v34 + 1;	// L80
          int v36 = v35;	// L81
          int32_t v37 = v1[v36][v11];	// L82
          v1[v33][v11] = v37;	// L84
        }
      }
      int v38 = i;	// L94
      ap_int<34> v39 = v38;	// L95
      ap_int<34> v40 = v39 + 1;	// L98
      bool v41 = v40 < 180;	// L101
      bool v42 = v14 & v41;	// L102
      if (v42) {	// L103
        int v43 = i;	// L108
        ap_int<34> v44 = v43;	// L109
        bool v45 = v44 < v13;	// L110
        if (v45) {	// L111
          int v46 = i;	// L112
          int32_t v47 = v0[v46];	// L113
          int32_t v48 = v0[v11];	// L114
          ap_int<33> v49 = v47;	// L115
          ap_int<33> v50 = v48;	// L116
          ap_int<33> v51 = v49 + v50;	// L117
          int32_t v52 = v51;	// L118
          int32_t w;	// L119
          w = v52;	// L120
          int32_t match;	// L121
          match = 0;	// L123
          int32_t v55 = w;	// L125
          bool v56 = v55 == 3;	// L126
          if (v56) {	// L127
            match = 1;	// L129
          }
          int32_t s2;	// L131
          s2 = 0;	// L133
          int v58 = i;	// L134
          ap_int<34> v59 = v58;	// L135
          ap_int<34> v60 = v59 + 1;	// L138
          int v61 = v60;	// L139
          int v62 = v13;	// L144
          int32_t v63 = v1[v61][v62];	// L145
          ap_int<33> v64 = v63;	// L146
          int32_t v65 = match;	// L147
          ap_int<33> v66 = v65;	// L148
          ap_int<33> v67 = v64 + v66;	// L149
          int32_t v68 = v67;	// L150
          s2 = v68;	// L151
          int v69 = i;	// L152
          int32_t v70 = v1[v69][v11];	// L153
          int32_t v71 = s2;	// L154
          bool v72 = v70 < v71;	// L155
          if (v72) {	// L156
            int v73 = i;	// L157
            int32_t v74 = s2;	// L158
            v1[v73][v11] = v74;	// L159
          }
        } else {
          int v75 = i;	// L162
          int32_t v76 = v1[v75][v11];	// L163
          ap_int<34> v77 = v75;	// L165
          ap_int<34> v78 = v77 + 1;	// L168
          int v79 = v78;	// L169
          int v80 = v13;	// L174
          int32_t v81 = v1[v79][v80];	// L175
          bool v82 = v76 < v81;	// L176
          if (v82) {	// L177
            int v83 = i;	// L178
            ap_int<34> v84 = v83;	// L179
            ap_int<34> v85 = v84 + 1;	// L182
            int v86 = v85;	// L183
            int32_t v87 = v1[v86][v80];	// L189
            v1[v83][v11] = v87;	// L191
          }
        }
      }
      int v88 = i;	// L195
      ap_int<34> v89 = v88;	// L196
      ap_int<34> v90 = v89 + 1;	// L199
      int v91 = v90;	// L200
      for (int v92 = v91; v92 < v11; v92 += 1) {	// L203
        int v93 = i;	// L204
        int32_t v94 = v1[v93][v92];	// L205
        ap_int<34> v95 = v92;	// L206
        ap_int<34> v96 = v95 + 1;	// L209
        int v97 = v96;	// L210
        int32_t v98 = v1[v97][v11];	// L211
        ap_int<33> v99 = v94;	// L212
        ap_int<33> v100 = v98;	// L213
        ap_int<33> v101 = v99 + v100;	// L214
        int32_t v102 = v101;	// L215
        int32_t s3;	// L216
        s3 = v102;	// L217
        int v104 = i;	// L218
        int32_t v105 = v1[v104][v11];	// L219
        int32_t v106 = s3;	// L220
        bool v107 = v105 < v106;	// L221
        if (v107) {	// L222
          int v108 = i;	// L223
          int32_t v109 = s3;	// L224
          v1[v108][v11] = v109;	// L225
        }
      }
    }
  }
}
