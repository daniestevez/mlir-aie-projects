// x-engine
//
// X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
//
// Copyright 2026 Daniel Estevez <daniel@destevez.net>
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <aie_api/aie.hpp>

template <int N, int M, int T>
  requires(N % 2 == 0)
static inline void multiply_conj_transpose_NxTx8x8_MxTx8x8(
    v64int8 *__restrict a, v64int8 *__restrict b, v64acc32 *__restrict out) {
  // a is N x T x complex x 8 x 8
  // b is M x T x complex x 8 x 8
  // out is N x M x complex x 8 x 8

  event0();

  for (int n = 0; n < N; n += 2) {
    v64int8 *__restrict pb = b;
    for (int m = 0; m < M; ++m) {
      v64acc32 out0_re = out[0];
      v64acc32 out0_im = out[1];
      v64acc32 out1_re = out[2 * M];
      v64acc32 out1_im = out[2 * M + 1];

      v64int8 *__restrict pa0 = a;
      v64int8 *__restrict pa1 = a + 2 * T;
      for (int t = 0; t < T; ++t) {
        v64int8 z0_re = *pa0++;
        v64int8 z0_im = *pa0++;
        v64int8 z1_re = *pa1++;
        v64int8 z1_im = *pa1++;
        v64int8 w_re = *pb++;
        v64int8 w_im = *pb++;
        // transpose w
        w_re = shuffle(w_re, w_re, T8_8x8);
        w_im = shuffle(w_im, w_im, T8_8x8);
        out0_re = mac_8x8_8x8(z0_im, w_im, mac_8x8_8x8(z0_re, w_re, out0_re));
        out0_im = mac_8x8_8x8(z0_im, w_re, msc_8x8_8x8(z0_re, w_im, out0_im));
        out1_re = mac_8x8_8x8(z1_im, w_im, mac_8x8_8x8(z1_re, w_re, out1_re));
        out1_im = mac_8x8_8x8(z1_im, w_re, msc_8x8_8x8(z1_re, w_im, out1_im));
      }

      out[0] = out0_re;
      out[1] = out0_im;
      out[2 * M] = out1_re;
      out[2 * M + 1] = out1_im;
      out += 2;
    }

    a += 4 * T;
    out += 2 * M;
  }

  event1();
}

template <int N, int M>
static inline void set_zero_complex_acc_NxMx8x8(v64acc32 *__restrict acc) {
  // The factor 2 is because the data is complex
  for (int j = 0; j < 2 * N * M; ++j) {
    *acc++ = {};
  }
}

static constexpr int kN = 8;
static constexpr int kM = 4;
static constexpr int kT = 15;

extern "C" {

void x_engine_kernel(v64int8 *__restrict a, v64int8 *__restrict b,
                     v64acc32 *__restrict out) {
  multiply_conj_transpose_NxTx8x8_MxTx8x8<kN, kM, kT>(a, b, out);
}

void x_engine_acc_zero(v64acc32 *__restrict acc) {
  set_zero_complex_acc_NxMx8x8<kN, kM>(acc);
}

} // extern "C"
