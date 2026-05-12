# x-engine
#
# X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
#
# This file contains a benchmark of the throughput for ingesting data into the
# NPU.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import argparse
import time

import numpy as np
from aie import iron
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2


@iron.jit
def ingest_throughput_benchmark(
    data: In,
    *,
    data_size: CompileTime[int],
    chunk_size: CompileTime[int],
    num_channels: CompileTime[int],
    num_columns: CompileTime[int],
):
    num_blocks = num_columns * num_channels
    assert data_size % (num_blocks * chunk_size) == 0
    block_size = data_size // num_blocks
    chunks_per_block = block_size // chunk_size
    data_ty = np.ndarray[(data_size,), np.dtype[np.uint8]]
    chunk_ty = np.ndarray[(chunk_size,), np.dtype[np.uint8]]
    sync_ty = np.ndarray[(1,), np.dtype[np.uint32]]
    in_fifos = [
        [
            ObjectFifo(chunk_ty, name=f"in_fifo_col{col}_chan{chan}")
            for chan in range(num_channels)
        ]
        for col in range(num_columns)
    ]
    termination_fifos = [
        [
            ObjectFifo(sync_ty, depth=1, name=f"termination_fifo_col{col}_chan{chan}")
            for chan in range(num_channels)
        ]
        for col in range(num_columns)
    ]

    def core_fn(in_fifo, termination):
        termination.acquire(1)
        for _ in range_(chunks_per_block):
            in_fifo.acquire(1)
            in_fifo.release(1)
        termination.release(1)

    workers = [
        Worker(
            core_fn, [in_fifos[col][chan].cons(), termination_fifos[col][chan].prod()]
        )
        for col in range(num_columns)
        for chan in range(num_channels)
    ]

    taps = [
        [
            TensorAccessPattern(
                (1, data_size),
                block_size * (num_channels * col + chan),
                [1, 1, 1, block_size],
                [0, 0, 0, 1],
            )
            for chan in range(num_channels)
        ]
        for col in range(num_columns)
    ]

    rt = Runtime()
    with rt.sequence(data_ty, sync_ty) as (data_in, termination):
        rt.start(*workers)
        for col in range(num_columns):
            for chan in range(num_channels):
                rt.fill(in_fifos[col][chan].prod(), data_in, taps[col][chan])
        for col in range(num_columns):
            for chan in range(num_channels):
                rt.drain(
                    termination_fifos[col][chan].cons(),
                    termination,
                    wait=True,
                )

    return Program(NPU2(), rt).resolve_program()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-channels",
        default=2,
        type=int,
        help="Number of shimDMA channels per shimNOC tile [default=%(default)r]",
    )
    parser.add_argument(
        "--num-columns",
        default=8,
        type=int,
        help="Number of NPU columns [default=%(default)r]",
    )
    parser.add_argument(
        "--chunk-size",
        default=4096,
        type=int,
        help="Chunk size in bytes [default=%(default)r]",
    )
    parser.add_argument(
        "--data-size",
        default=256 * 2**20,
        type=int,
        help="Data size in bytes [default=%(default)r]",
    )
    parser.add_argument(
        "--benchmark-iterations",
        default=1000,
        type=int,
        help="Number of benchmark iterations [default=%(default)r]",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = iron.arange(args.data_size, dtype=np.uint8, device="npu")

    def kernel():
        ingest_throughput_benchmark(
            data,
            data_size=args.data_size,
            chunk_size=args.chunk_size,
            num_channels=args.num_channels,
            num_columns=args.num_columns,
        )

    # call to JIT-compile kernel
    kernel()

    # benchmark calls
    total_start = time.perf_counter()
    elapsed_per_kernel = np.empty(args.benchmark_iterations)
    for j in range(args.benchmark_iterations):
        kernel_start = time.perf_counter()
        kernel()
        elapsed_per_kernel[j] = time.perf_counter() - kernel_start
    total_elapsed = time.perf_counter() - total_start

    bandwidth = args.benchmark_iterations * args.data_size / total_elapsed
    print(f"Total elapsed time: {total_elapsed:.3f} s")
    print(
        f"Average bandwidth: {bandwidth * 1e-9:.3f} GB/s, {bandwidth * 8 * 1e-9:.3f} Gbps"
    )
    print(
        "Per kernel min/avg/max: "
        f"{np.min(elapsed_per_kernel) * 1e3:.3f}/"
        f"{np.average(elapsed_per_kernel) * 1e3:.3f}/"
        f"{np.max(elapsed_per_kernel) * 1e3:.3f} ms"
    )
    bandwidth_kernel = args.data_size / elapsed_per_kernel
    print(
        "Per kernel min/avg/max: "
        f"{np.min(bandwidth_kernel) * 1e-9:.3f}/"
        f"{np.average(bandwidth_kernel) * 1e-9:.3f}/"
        f"{np.max(bandwidth_kernel) * 1e-9:.3f} GB/s"
    )


if __name__ == "__main__":
    main()
