# peak-tops
#
# This project demonstrates the advertised peak TOPS performance on AIE-ML
# engines.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import argparse

import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils import DefaultNPURuntime
from aie.utils.trace import TraceConfig
from aie.utils.npukernel import NPUKernel
import numpy as np


def main(args):
    sync_buffer = iron.zeros([1], dtype=np.uint32)
    if args.trace_size:
        trace_config = TraceConfig(args.trace_size, args.trace_file)
    else:
        trace_config = None
    npu_kernel = NPUKernel(args.xclbin, args.instr, trace_config)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)
    buffers = [sync_buffer, sync_buffer]
    if args.trace_size:
        buffers = DefaultNPURuntime.prepare_args_for_trace(buffers, trace_config)
    res = DefaultNPURuntime.run(kernel_handle, buffers)
    if args.trace_size:
        trace_buffer, ctrl_buffer = DefaultNPURuntime.extract_trace_from_args(
            buffers, trace_config
        )
        DefaultNPURuntime.process_trace(trace_buffer, ctrl_buffer, trace_config)
    npu_time = res.npu_time * 1e-9
    # Obtained from the trace profile and study of the asm by hand
    cycles_per_iteration = 539
    num_matrices_per_iteration = 2 * 16384 / (8 * 8)
    macs_per_iteration = num_matrices_per_iteration * 8 * 8 * 8
    ops = args.compute_tiles * 2 * macs_per_iteration * args.iterations / npu_time
    compute_clock = args.iterations * cycles_per_iteration / npu_time
    print(f"elapsed time: {npu_time:.3f} seconds")
    print(f"compute core clock rate: {compute_clock * 1e-9:.3f} GHz")
    print(f"TOPS: {ops * 1e-12:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument(
        "--compute-tiles", default=32, type=int, help="Number of compute tiles"
    )
    parser.add_argument(
        "--iterations", default=2**22, type=int, help="Number of iterations"
    )
    args = parser.parse_args()
    main(args)
