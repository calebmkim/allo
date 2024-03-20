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
void kernel_adi(
  int32_t v0[5][5],
  int32_t v1[5][5],
  int32_t v2[5][5],
  int32_t v3[5][5]
) {	// L2
  l_S_t_0_t: for (int t = 0; t < 4; t++) {	// L3
    l_S_i_0_i: for (int i = 0; i < 3; i++) {	// L4
      int v6 = (i + 1);	// L4
      v1[0][v6] = 1;	// L7
      v2[v6][0] = 0;	// L10
      int32_t v7 = v1[0][v6];	// L11
      v3[v6][0] = v7;	// L12
      l_S_j_0_j: for (int j = 0; j < 3; j++) {	// L13
        int v9 = (j + 1);	// L13
        int32_t v10 = v2[v6][(v9 - 1)];	// L18
        int64_t v11 = v10;	// L21
        int64_t v12 = v11 * -6;	// L22
        ap_int<65> v13 = v12;	// L23
        ap_int<65> v14 = v13 + 13;	// L26
        int32_t v15 = v14;	// L27
        int32_t v16 = 6 / v15;	// L28
        v2[v6][v9] = v16;	// L29
        int32_t v17 = v0[v9][(v6 - 1)];	// L34
        int64_t v18 = v17;	// L36
        int64_t v19 = v18 * 3;	// L37
        int32_t v20 = v0[v9][v6];	// L47
        ap_int<97> v21 = v20;	// L49
        ap_int<97> v22 = v21 * -5;	// L50
        ap_int<98> v23 = v19;	// L51
        ap_int<98> v24 = v22;	// L52
        ap_int<98> v25 = v23 + v24;	// L53
        int32_t v26 = v0[v9][(v6 + 1)];	// L54
        int64_t v27 = v26;	// L57
        int64_t v28 = v27 * -3;	// L58
        ap_int<99> v29 = v25;	// L59
        ap_int<99> v30 = v28;	// L60
        ap_int<99> v31 = v29 - v30;	// L61
        int32_t v32 = v3[v6][(v9 - 1)];	// L62
        int64_t v33 = v32;	// L65
        int64_t v34 = v33 * -6;	// L66
        ap_int<100> v35 = v31;	// L67
        ap_int<100> v36 = v34;	// L68
        ap_int<100> v37 = v35 - v36;	// L69
        int32_t v38 = v37;	// L70
        int32_t v39 = v2[v6][(v9 - 1)];	// L71
        int64_t v40 = v39;	// L74
        int64_t v41 = v40 * -6;	// L75
        ap_int<65> v42 = v41;	// L76
        ap_int<65> v43 = v42 + 13;	// L79
        int32_t v44 = v43;	// L80
        int32_t v45 = v38 / v44;	// L81
        v3[v6][v9] = v45;	// L82
      }
      v1[4][v6] = 1;	// L86
      l_S_j_rev_1_j_rev: for (int j_rev = 0; j_rev < 4; j_rev++) {	// L87
        ap_int<34> v47 = j_rev;	// L94
        ap_int<34> v48 = 3 - v47;	// L95
        int v49 = v48;	// L96
        int j1;	// L97
        j1 = v49;	// L98
        int v51 = j1;	// L99
        int32_t v52 = v2[v6][v51];	// L100
        ap_int<34> v53 = v51;	// L102
        ap_int<34> v54 = v53 + 1;	// L105
        int v55 = v54;	// L106
        int32_t v56 = v1[v55][v6];	// L107
        int64_t v57 = v52;	// L108
        int64_t v58 = v56;	// L109
        int64_t v59 = v57 * v58;	// L110
        int32_t v60 = v3[v6][v51];	// L112
        ap_int<65> v61 = v59;	// L113
        ap_int<65> v62 = v60;	// L114
        ap_int<65> v63 = v61 + v62;	// L115
        int32_t v64 = v63;	// L116
        v1[v51][v6] = v64;	// L118
      }
    }
    l_S_i_3_i1: for (int i1 = 0; i1 < 3; i1++) {	// L121
      int v66 = (i1 + 1);	// L121
      v0[v66][0] = 1;	// L124
      v2[v66][0] = 0;	// L127
      int32_t v67 = v0[v66][0];	// L128
      v3[v66][0] = v67;	// L129
      l_S_j_3_j2: for (int j2 = 0; j2 < 3; j2++) {	// L130
        int v69 = (j2 + 1);	// L130
        int32_t v70 = v2[v66][(v69 - 1)];	// L135
        int64_t v71 = v70;	// L138
        int64_t v72 = v71 * -3;	// L139
        ap_int<65> v73 = v72;	// L140
        ap_int<65> v74 = v73 + 7;	// L143
        int32_t v75 = v74;	// L144
        int32_t v76 = 3 / v75;	// L145
        v2[v66][v69] = v76;	// L146
        int32_t v77 = v1[(v66 - 1)][v69];	// L151
        int64_t v78 = v77;	// L153
        int64_t v79 = v78 * 6;	// L154
        int32_t v80 = v1[v66][v69];	// L164
        ap_int<97> v81 = v80;	// L166
        ap_int<97> v82 = v81 * -11;	// L167
        ap_int<98> v83 = v79;	// L168
        ap_int<98> v84 = v82;	// L169
        ap_int<98> v85 = v83 + v84;	// L170
        int32_t v86 = v1[(v66 + 1)][v69];	// L171
        int64_t v87 = v86;	// L174
        int64_t v88 = v87 * -6;	// L175
        ap_int<99> v89 = v85;	// L176
        ap_int<99> v90 = v88;	// L177
        ap_int<99> v91 = v89 - v90;	// L178
        int32_t v92 = v3[v66][(v69 - 1)];	// L179
        int64_t v93 = v92;	// L182
        int64_t v94 = v93 * -3;	// L183
        ap_int<100> v95 = v91;	// L184
        ap_int<100> v96 = v94;	// L185
        ap_int<100> v97 = v95 - v96;	// L186
        int32_t v98 = v2[v66][(v69 - 1)];	// L187
        int64_t v99 = v98;	// L190
        int64_t v100 = v99 * -3;	// L191
        ap_int<65> v101 = v100;	// L192
        ap_int<65> v102 = v101 + 7;	// L195
        ap_int<100> v103 = v102;	// L196
        ap_int<100> v104 = v97 / v103;	// L197
        int32_t v105 = v104;	// L198
        v3[v66][v69] = v105;	// L199
      }
      v0[v66][4] = 1;	// L203
      l_S_j_rev_4_j_rev1: for (int j_rev1 = 0; j_rev1 < 4; j_rev1++) {	// L204
        ap_int<34> v107 = j_rev1;	// L211
        ap_int<34> v108 = 3 - v107;	// L212
        int v109 = v108;	// L213
        int j3;	// L214
        j3 = v109;	// L215
        int v111 = j3;	// L216
        int32_t v112 = v2[v66][v111];	// L217
        ap_int<34> v113 = v111;	// L219
        ap_int<34> v114 = v113 + 1;	// L222
        int v115 = v114;	// L223
        int32_t v116 = v0[v66][v115];	// L224
        int64_t v117 = v112;	// L225
        int64_t v118 = v116;	// L226
        int64_t v119 = v117 * v118;	// L227
        int32_t v120 = v3[v66][v111];	// L229
        ap_int<65> v121 = v119;	// L230
        ap_int<65> v122 = v120;	// L231
        ap_int<65> v123 = v121 + v122;	// L232
        int32_t v124 = v123;	// L233
        v0[v66][v111] = v124;	// L235
      }
    }
  }
}
