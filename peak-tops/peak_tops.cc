// peak-tops
//
// This project demonstrates the advertised peak TOPS performance on AIE-ML
// engines.
//
// Copyright 2016 Daniel Estevez <daniel@destevez.net>
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <aie_api/aie.hpp>

extern "C" {

void peak_tops(v64int8 *__restrict a, v64int8 *__restrict b0,
               v64int8 *__restrict b1, v64int8 *__restrict out) {
  event0();

  constexpr int N = 16384;
  constexpr int num_vectors = N / 64;

  v64acc32 acc0 = {};
  v64acc32 acc1 = {};

  for (int i = 0; i < num_vectors; ++i) {
    v64int8 x = *a++;
    acc0 = mac_8x8_8x8(x, *b0++, acc0);
    acc1 = mac_8x8_8x8(x, *b1++, acc1);
  }
  out[0] = ssrs(acc0, 0);
  out[1] = ssrs(acc1, 0);

  event1();
}
}
