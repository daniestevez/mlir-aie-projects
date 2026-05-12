# peak-tops
#
# This project demonstrates the advertised peak TOPS performance on AIE-ML
# engines.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import argparse

from aie.iron import Kernel, Program, Runtime, Worker, ObjectFifo, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
import numpy as np


def peak_tops(trace_size, num_compute_tiles, num_compute_per_column, iterations):
    N = 16384
    vector_size = 64
    buff_ty = np.ndarray[(N,), np.dtype[np.int8]]
    out_ty = np.ndarray[(2 * vector_size,), np.dtype[np.int8]]
    a_buffs = [
        Buffer(buff_ty, name=f"a_buff_core{n:02}") for n in range(num_compute_tiles)
    ]
    b0_buffs = [
        Buffer(buff_ty, name=f"b0_buff_core{n:02}") for n in range(num_compute_tiles)
    ]
    b1_buffs = [
        Buffer(buff_ty, name=f"b1_buff_core{n:02}") for n in range(num_compute_tiles)
    ]
    out_buffs = [
        Buffer(out_ty, name=f"out_buff_core{n:02}") for n in range(num_compute_tiles)
    ]

    kernel = Kernel(
        "peak_tops",
        "build/peak_tops.o",
        [buff_ty, buff_ty, buff_ty, out_ty],
    )

    sync_ty = np.ndarray[(1,), np.dtype[np.uint32]]
    start_fifo = ObjectFifo(sync_ty, depth=1, name="start_fifo")
    num_columns = (
        num_compute_tiles + num_compute_per_column - 1
    ) // num_compute_per_column
    column_termination_fifos = [
        ObjectFifo(sync_ty, depth=1, name=f"termination_fifo_col{n}")
        for n in range(num_columns)
    ]
    worker_termination_fifos = []
    for n, fifo in enumerate(column_termination_fifos):
        workers_this_column = min(
            num_compute_per_column, num_compute_tiles - len(worker_termination_fifos)
        )
        worker_termination_fifos.extend(
            fifo.prod().join(
                [0] * workers_this_column,
                obj_types=[sync_ty] * workers_this_column,
                names=[
                    f"termination_fifo_col{n}_core{m:02}"
                    for m in range(
                        len(worker_termination_fifos),
                        len(worker_termination_fifos) + workers_this_column,
                    )
                ],
                tile=Tile(n, 1),
            )
        )

    def core_fn(start, termination, a, b0, b1, out, kernel):
        start.acquire(1)
        start.release(1)
        for _ in range_(iterations):
            kernel(a, b0, b1, out)
        termination.acquire(1)
        termination.release(1)

    enable_trace = trace_size > 0
    workers = [
        Worker(
            core_fn,
            [
                start_fifo.cons(),
                worker_termination_fifos[n].prod(),
                a_buffs[n],
                b0_buffs[n],
                b1_buffs[n],
                out_buffs[n],
                kernel,
            ],
            tile=Tile(n // num_compute_per_column, (n % num_compute_per_column) + 2),
        )
        for n in range(num_compute_tiles)
    ]

    rt = Runtime()
    with rt.sequence(sync_ty, sync_ty) as (start, termination):
        if enable_trace:
            rt.enable_trace(trace_size, workers=workers)
        rt.start(*workers)
        rt.fill(start_fifo.prod(), start, tile=Tile(0, 0))
        for n, fifo in enumerate(column_termination_fifos):
            rt.drain(fifo.cons(), termination, wait=True, tile=Tile(n, 0))

    return Program(NPU2(), rt).resolve_program()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-size", default=0, type=int, help="Trace size")
    parser.add_argument(
        "--compute-tiles", default=32, type=int, help="Number of compute tiles"
    )
    parser.add_argument(
        "--compute-per-column",
        default=4,
        type=int,
        help="Number of compute tiles to use per column",
    )
    parser.add_argument(
        "--iterations", default=2**22, type=int, help="Number of iterations"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    program = peak_tops(
        args.trace_size, args.compute_tiles, args.compute_per_column, args.iterations
    )
    print(program)


if __name__ == "__main__":
    main()
