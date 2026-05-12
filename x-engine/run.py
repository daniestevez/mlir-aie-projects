# x-engine
#
# X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import argparse

import numpy as np
from aie import iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig

from x_engine import Dimensions


def main(args):
    dimensions = Dimensions()
    a_buffer = iron.randint(
        low=-128, high=127, size=dimensions.input_length(), dtype=np.int8
    )
    b_buffer = iron.randint(
        low=-128, high=127, size=dimensions.input_length(), dtype=np.int8
    )
    acc_buffer = iron.zeros(dimensions.all_acc_length(), dtype=np.int32)
    if args.trace_size:
        trace_config = TraceConfig(args.trace_size, args.trace_file)
    else:
        trace_config = None
    npu_kernel = NPUKernel(args.xclbin, args.instr, trace_config)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)
    buffers = [a_buffer, b_buffer, acc_buffer]
    if args.trace_size:
        buffers = DefaultNPURuntime.prepare_args_for_trace(buffers, trace_config)
        DefaultNPURuntime.run(kernel_handle, buffers)
        trace_buffer, ctrl_buffer = DefaultNPURuntime.extract_trace_from_args(
            buffers, trace_config
        )
        DefaultNPURuntime.process_trace(trace_buffer, ctrl_buffer, trace_config)
    else:
        npu_times = []
        num_runs = 100
        for _ in range(num_runs):
            res = DefaultNPURuntime.run(kernel_handle, buffers)
            npu_times.append(res.npu_time * 1e-9)
        npu_time = np.average(npu_times)
        print(
            "elapsed time: "
            f"{npu_time * 1e3:.3f}/{np.min(npu_times) * 1e3:.3f}/{np.max(npu_times) * 1e3:.3f} "
            f"avg/min/max ms"
        )
        num_samples = (
            dimensions.N_PFB * dimensions.samples_per_packet() * dimensions.integrations
        )
        throughput = num_samples / npu_time
        print(f"throughput: {throughput * 1e-6:.3f} Msps")
        # the factor 4 is due to complex multiplication
        macs = 4 * dimensions.num_streams() ** 2 * num_samples
        tops = macs / npu_time * 1e-12
        print(f"TOPS: {tops:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    args = parser.parse_args()
    main(args)
