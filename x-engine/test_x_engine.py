# x-engine
#
# X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import random

import numpy as np
from aie import iron
from aie.utils import DefaultNPURuntime
from aie.utils.npukernel import NPUKernel

from x_engine import Dimensions


class TestXEngine:
    def make_buffers(self):
        self.dimensions = Dimensions()
        self.a_buffer = iron.randint(
            low=-128, high=127, size=self.dimensions.input_length(), dtype=np.int8
        )
        self.b_buffer = iron.randint(
            low=-128, high=127, size=self.dimensions.input_length(), dtype=np.int8
        )
        self.acc_buffer = iron.randint(
            low=-(2**31),
            high=2**31 - 1,
            size=self.dimensions.all_acc_length(),
            dtype=np.int32,
        )

    def run_kernel(self):
        # load the kernel only once
        if not hasattr(self, "kernel_handle"):
            npu_kernel = NPUKernel("build/x_engine.xclbin", "build/x_engine_insts.bin")
            self.kernel_handle = DefaultNPURuntime.load(npu_kernel)
        buffers = [self.a_buffer, self.b_buffer, self.acc_buffer]
        DefaultNPURuntime.run(self.kernel_handle, buffers)
        # ensure that we have a coherent view of the output from the CPU
        self.acc_buffer.to("cpu")

    def cleanup(self):
        del self.a_buffer
        del self.b_buffer
        del self.acc_buffer

    def test_zeros_input(self):
        self.make_buffers()
        self.a_buffer[:] = 0
        self.b_buffer[:] = 0
        self.run_kernel()
        assert np.all(self.acc_buffer == np.int32(0))
        self.cleanup()

    def test_a_zeros_input(self):
        self.make_buffers()
        self.a_buffer[:] = 0
        self.run_kernel()
        assert np.all(self.acc_buffer == np.int32(0))
        self.cleanup()

    def test_b_zeros_input(self):
        self.make_buffers()
        self.b_buffer[:] = 0
        self.run_kernel()
        assert np.all(self.acc_buffer == np.int32(0))
        self.cleanup()

    def test_ones_input(self):
        self.make_buffers()
        self.a_buffer[:] = 0
        self.a_buffer.data.reshape(-1, self.dimensions.simd_size())[::2] = 1
        self.b_buffer[:] = self.a_buffer
        self.run_kernel()
        n_accs = self.dimensions.samples_per_packet() * self.dimensions.integrations
        acc_re = self.acc_buffer.data.reshape(-1, self.dimensions.vmac_size())[::2]
        acc_im = self.acc_buffer.data.reshape(-1, self.dimensions.vmac_size())[1::2]
        assert np.all(acc_re == np.int32(n_accs))
        assert np.all(acc_im == np.int32(0))
        self.cleanup()

    def test_single_one_input(self):
        self.make_buffers()
        packet = random.randrange(0, self.dimensions.integrations)
        stream_a = random.randrange(0, self.dimensions.num_streams())
        stream_b = random.randrange(0, self.dimensions.num_streams())
        pfb_channel = random.randrange(0, self.dimensions.N_PFB)
        sample_block = random.randrange(0, self.dimensions.num_sample_blocks)
        sample_word = random.randrange(0, self.dimensions.T)
        sample = random.randrange(0, self.dimensions.simd_size())
        self.a_buffer[:] = 0
        self.b_buffer[:] = 0

        def index(s):
            return sample + 2 * self.dimensions.simd_size() * (
                sample_word
                + self.dimensions.T
                * (
                    sample_block
                    + self.dimensions.num_sample_blocks
                    * (
                        pfb_channel
                        + self.dimensions.N_PFB
                        * (s + self.dimensions.num_streams() * packet)
                    )
                )
            )

        self.a_buffer[index(stream_a)] = 1
        self.b_buffer[index(stream_b)] = 1
        self.run_kernel()

        expected_idx = self.acc_index(pfb_channel, stream_a, stream_b)
        out_location = np.where(self.acc_buffer.data != 0)[0]
        assert len(out_location) == 1
        assert out_location[0] == expected_idx
        self.cleanup()

    def test_random_inputs(self):
        self.make_buffers()
        self.run_kernel()

        shape = (
            self.dimensions.integrations,
            self.dimensions.num_streams(),
            self.dimensions.N_PFB,
            self.dimensions.num_sample_blocks * self.dimensions.T,
            2,
            self.dimensions.simd_size(),
        )
        for pfb_channel in range(self.dimensions.N_PFB):
            for stream_a in range(self.dimensions.num_streams()):
                a_re = (
                    self.a_buffer.data.reshape(shape)[:, stream_a, pfb_channel, :, 0, :]
                    .ravel()
                    .astype("int32")
                )
                a_im = (
                    self.a_buffer.data.reshape(shape)[:, stream_a, pfb_channel, :, 1, :]
                    .ravel()
                    .astype("int32")
                )
                for stream_b in range(self.dimensions.num_streams()):
                    b_re = (
                        self.b_buffer.data.reshape(shape)[
                            :, stream_b, pfb_channel, :, 0, :
                        ]
                        .ravel()
                        .astype("int32")
                    )
                    b_im = (
                        self.b_buffer.data.reshape(shape)[
                            :, stream_b, pfb_channel, :, 1, :
                        ]
                        .ravel()
                        .astype("int32")
                    )
                    expected_re = np.sum(a_re * b_re + a_im * b_im)
                    expected_im = np.sum(a_im * b_re - a_re * b_im)
                    out_idx = self.acc_index(pfb_channel, stream_a, stream_b)
                    result_re = self.acc_buffer[out_idx]
                    result_im = self.acc_buffer[out_idx + self.dimensions.vmac_size()]
                    assert result_re == expected_re
                    assert result_im == expected_im

        self.cleanup()

    def acc_index(self, pfb_channel, stream_a, stream_b):
        c0 = stream_b // (self.dimensions.M * self.dimensions.simd_size())
        c1 = (
            stream_b - c0 * self.dimensions.M * self.dimensions.simd_size()
        ) // self.dimensions.simd_size()
        c2 = stream_b % self.dimensions.simd_size()
        r0 = stream_a // (self.dimensions.N * self.dimensions.simd_size())
        r1 = (
            stream_a - r0 * self.dimensions.N * self.dimensions.simd_size()
        ) // self.dimensions.simd_size()
        r2 = stream_a % self.dimensions.simd_size()
        return (
            pfb_channel * 2 * self.dimensions.num_streams() ** 2
            + c0 * self.dimensions.col_acc_length()
            + r0 * self.dimensions.acc_length()
            + r1 * self.dimensions.M * 2 * self.dimensions.vmac_size()
            + c1 * 2 * self.dimensions.vmac_size()
            + r2 * self.dimensions.simd_size()
            + c2
        )
