# x-engine
#
# X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
#
# Copyright 2026 Daniel Estevez <daniel@destevez.net>
# SPDX-License-Identifier: MIT OR Apache-2.0

import argparse

import aie.utils.trace as trace_utils
import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    EndOp,
    LockAction,
    ObjectFifoPort,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    external_func,
    flow,
    lock,
    mem,
    memtile_dma,
    next_bd,
    object_fifo,
    object_fifo_link,
    tile,
    use_lock,
)
from aie.dialects.aiex import (
    bds,
    dma_await_task,
    dma_configure_task,
    dma_start_task,
    runtime_sequence,
    shim_dma_bd,
    shim_dma_single_bd_task,
)
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.utils.trace.events import (
    CoreEvent,
    MemEvent,
    MemTileEvent,
    MemTilePortEvent,
    PortEvent,
    ShimTileEvent,
)


class Dimensions:
    def __init__(self):
        self.N = 8
        self.M = 4
        self.T = 15
        self.array_rows = 4
        self.array_cols = 8

        # Layout for input buffers
        #
        # packet (coarsest time) x stream (N x array_rows x 8) x
        # x PFB channel x sample blocks (medium-grained time)
        # x T sample words (finer time)
        # x real/imaginary
        # x 8 samples (finest time)

        self.N_PFB = 9
        self.num_sample_blocks = 4

        self.integrations = 1024

        assert self.N * self.array_rows == self.M * self.array_cols
        assert self.items_per_packet() <= 8800

    def a_length(self):
        return self.N * self.T * 2 * self.vmac_size()

    def b_length(self):
        return self.M * self.T * 2 * self.vmac_size()

    def acc_length(self):
        return self.N * self.M * 2 * self.vmac_size()

    def col_acc_length(self):
        return self.array_rows * self.acc_length()

    def all_acc_length(self):
        return self.N_PFB * self.array_rows * self.array_cols * self.acc_length()

    def input_length(self):
        return self.items_per_packet() * self.num_streams() * self.integrations

    def vmac_size(self):
        return self.simd_size() ** 2

    def simd_size(self):
        return 8

    def samples_per_block(self):
        return self.T * self.simd_size()

    def samples_per_packet(self):
        return self.samples_per_block() * self.num_sample_blocks

    def items_per_packet(self):
        return self.samples_per_packet() * 2 * self.N_PFB

    def num_streams(self):
        return self.N * self.array_rows * self.simd_size()


def x_engine(trace_size):
    dimensions = Dimensions()
    if trace_size:
        num_rows = dimensions.array_rows - 1
        num_cols = dimensions.array_cols
    else:
        num_rows = dimensions.array_rows
        num_cols = dimensions.array_cols
    num_mem_tiles = max(num_cols, num_rows)
    num_shim_tiles = max(num_cols, 2 * num_rows)
    a_ty = np.ndarray[(dimensions.a_length(),), np.dtype[np.int8]]
    block_a_ty = np.ndarray[
        (dimensions.num_sample_blocks * dimensions.a_length(),), np.dtype[np.int8]
    ]
    block_b_ty = np.ndarray[
        (dimensions.num_sample_blocks * dimensions.b_length(),), np.dtype[np.int8]
    ]
    b_ty = np.ndarray[(dimensions.b_length(),), np.dtype[np.int8]]
    acc_ty = np.ndarray[(dimensions.acc_length(),), np.dtype[np.int32]]
    col_acc_ty = np.ndarray[(dimensions.acc_length() * num_rows,), np.dtype[np.int32]]
    all_acc_ty = np.ndarray[(dimensions.all_acc_length(),), np.dtype[np.int32]]
    input_ty = np.ndarray[(dimensions.input_length(),), np.dtype[np.int8]]

    @device(AIEDevice.npu2)
    def device_body():
        shim_tiles = [tile(n, 0) for n in range(num_shim_tiles)]
        mem_tiles = [tile(n, 1) for n in range(num_mem_tiles)]
        compute_tiles = [
            [tile(col, 2 + row) for row in range(num_rows)] for col in range(num_cols)
        ]

        def a_mem_tile(row):
            return mem_tiles[2 * row]

        mem_a_buff_0 = [
            buffer(a_mem_tile(n), block_a_ty, name=f"mem_a_buff_0_row{n}")
            for n in range(num_rows)
        ]
        mem_a_buff_1 = [
            buffer(a_mem_tile(n), block_a_ty, name=f"mem_a_buff_1_row{n}")
            for n in range(num_rows)
        ]
        mem_a00_cons_prod_lock = [
            lock(
                a_mem_tile(n),
                init=1,
                sym_name=f"mem_a00_cons_prod_lock_row{n}",
            )
            for n in range(num_rows)
        ]
        mem_a01_cons_prod_lock = [
            lock(
                a_mem_tile(n),
                init=1,
                sym_name=f"mem_a01_cons_prod_lock_row{n}",
            )
            for n in range(num_rows)
        ]
        mem_a10_cons_prod_lock = [
            lock(
                a_mem_tile(n),
                init=1,
                sym_name=f"mem_a1_cons_prod_lock_row{n}",
            )
            for n in range(num_rows)
        ]
        mem_a11_cons_prod_lock = [
            lock(
                a_mem_tile(n),
                init=1,
                sym_name=f"mem_a11_cons_prod_lock_row{n}",
            )
            for n in range(num_rows)
        ]
        mem_a00_cons_cons_lock = [
            lock(a_mem_tile(n), init=0, sym_name=f"mem_a00_cons_cons_lock_row{n}")
            for n in range(num_rows)
        ]
        mem_a01_cons_cons_lock = [
            lock(a_mem_tile(n), init=0, sym_name=f"mem_a01_cons_cons_lock_row{n}")
            for n in range(num_rows)
        ]
        mem_a10_cons_cons_lock = [
            lock(a_mem_tile(n), init=0, sym_name=f"mem_a10_cons_cons_lock_row{n}")
            for n in range(num_rows)
        ]
        mem_a11_cons_cons_lock = [
            lock(a_mem_tile(n), init=0, sym_name=f"mem_a11_cons_cons_lock_row{n}")
            for n in range(num_rows)
        ]

        a_buff_0 = [
            [
                buffer(compute_tiles[n][m], a_ty, name=f"a_buff_0_col{n}_row{m}")
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        a_buff_1 = [
            [
                buffer(compute_tiles[n][m], a_ty, name=f"a_buff_1_col{n}_row{m}")
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        a_cons_prod_lock = [
            [
                lock(
                    compute_tiles[n][m],
                    init=2,
                    sym_name=f"a_cons_prod_lock_col{n}_row{m}",
                )
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        a_cons_cons_lock = [
            [
                lock(
                    compute_tiles[n][m],
                    init=0,
                    sym_name=f"a_cons_cons_lock_col{n}_row{m}",
                )
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]

        for row in range(num_rows):
            for ch in range(2):
                flow(
                    shim_tiles[2 * row + ch],
                    WireBundle.DMA,
                    0,
                    a_mem_tile(row),
                    WireBundle.DMA,
                    ch,
                )
        for row in range(num_rows):
            for col in range(num_cols):
                flow(
                    a_mem_tile(row),
                    WireBundle.DMA,
                    0,
                    compute_tiles[col][row],
                    WireBundle.DMA,
                    0,
                )

        def b_mem_tile(col):
            return mem_tiles[2 * (col // 2) + 1]

        mem_b_buff_0 = [
            buffer(b_mem_tile(n), block_b_ty, name=f"mem_b_buff_0_col{n}")
            for n in range(num_cols)
        ]
        mem_b_buff_1 = [
            buffer(b_mem_tile(n), block_b_ty, name=f"mem_b_buff_1_col{n}")
            for n in range(num_cols)
        ]
        mem_b_cons_prod_lock = [
            lock(
                b_mem_tile(n),
                init=2 * dimensions.num_sample_blocks,
                sym_name=f"mem_b_cons_prod_lock_col{n}",
            )
            for n in range(num_cols)
        ]
        mem_b_cons_cons_lock = [
            lock(b_mem_tile(n), init=0, sym_name=f"mem_b_cons_cons_lock_col{n}")
            for n in range(num_cols)
        ]

        b_buff_0 = [
            [
                buffer(compute_tiles[n][m], b_ty, name=f"b_buff_0_col{n}_row{m}")
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        b_buff_1 = [
            [
                buffer(compute_tiles[n][m], b_ty, name=f"b_buff_1_col{n}_row{m}")
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        b_cons_prod_lock = [
            [
                lock(
                    compute_tiles[n][m],
                    init=2,
                    sym_name=f"b_cons_prod_lock_col{n}_row{m}",
                )
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]
        b_cons_cons_lock = [
            [
                lock(
                    compute_tiles[n][m],
                    init=0,
                    sym_name=f"b_cons_cons_lock_col{n}_row{m}",
                )
                for m in range(num_rows)
            ]
            for n in range(num_cols)
        ]

        for col in range(num_cols):
            flow(
                shim_tiles[col],
                WireBundle.DMA,
                1,
                b_mem_tile(col),
                WireBundle.DMA,
                col % 2,
            )
            for row in range(num_rows):
                flow(
                    b_mem_tile(col),
                    WireBundle.DMA,
                    col % 2,
                    compute_tiles[col][row],
                    WireBundle.DMA,
                    1,
                )

        # memtile a buffer layout:
        # N x (sample_blocks x T) x real/imaginary x (8 x 8)
        a_in_dimensions = [
            (
                dimensions.N // 2,
                dimensions.num_sample_blocks * dimensions.a_length() // dimensions.N,
            ),
            (dimensions.simd_size(), dimensions.simd_size()),
            (2 * dimensions.T * dimensions.num_sample_blocks, dimensions.vmac_size()),
            (dimensions.simd_size(), 1),
        ]
        a_out_dimensions = [
            (
                dimensions.N,
                dimensions.num_sample_blocks * dimensions.a_length() // dimensions.N,
            ),
            # these two dimensions are not linearized because the max size of a dimension is 1023
            (dimensions.T, 2 * dimensions.vmac_size()),
            (2 * dimensions.vmac_size(), 1),
        ]
        a_out_step = 2 * dimensions.vmac_size() * dimensions.T

        def make_a_memtile(row):
            @memtile_dma(a_mem_tile(row))
            def mem_body_mem_a(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(
                        mem_a00_cons_prod_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.a_length() // 2,
                        dimensions=a_in_dimensions,
                    )
                    use_lock(
                        mem_a00_cons_cons_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[2])
                with block[2]:
                    use_lock(
                        mem_a10_cons_prod_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.a_length() // 2,
                        dimensions=a_in_dimensions,
                    )
                    use_lock(
                        mem_a10_cons_cons_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[1])
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(
                        mem_a01_cons_prod_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=dimensions.num_sample_blocks
                        * dimensions.a_length()
                        // 2,
                        len=dimensions.num_sample_blocks * dimensions.a_length() // 2,
                        dimensions=a_in_dimensions,
                    )
                    use_lock(
                        mem_a01_cons_cons_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[5])
                with block[5]:
                    use_lock(
                        mem_a11_cons_prod_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=dimensions.num_sample_blocks
                        * dimensions.a_length()
                        // 2,
                        len=dimensions.num_sample_blocks * dimensions.a_length() // 2,
                        dimensions=a_in_dimensions,
                    )
                    use_lock(
                        mem_a11_cons_cons_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[4])
                with block[6]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[19])
                with block[7]:
                    use_lock(
                        mem_a00_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    # dummy BD
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=0,
                    )
                    # dummy release
                    use_lock(mem_a00_cons_cons_lock[row], LockAction.Release, value=0)
                    next_bd(block[8])
                with block[8]:
                    use_lock(
                        mem_a01_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    # dummy release
                    use_lock(
                        mem_a01_cons_prod_lock[row],
                        LockAction.Release,
                        value=0,
                    )
                    next_bd(block[9])
                with block[9]:
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    next_bd(block[10])
                with block[10]:
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=2 * a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    next_bd(block[11])
                with block[11]:
                    # dummy acquire
                    use_lock(
                        mem_a00_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=0,
                    )
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=3 * a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    use_lock(
                        mem_a00_cons_prod_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[12])
                with block[12]:
                    # dummy acquire
                    use_lock(
                        mem_a01_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=0,
                    )
                    # dummy BD
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=0,
                    )
                    use_lock(
                        mem_a01_cons_prod_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[13])
                with block[13]:
                    use_lock(
                        mem_a10_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    # dummy BD
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=0,
                    )
                    # dummy release
                    use_lock(mem_a10_cons_cons_lock[row], LockAction.Release, value=0)
                    next_bd(block[14])
                with block[14]:
                    use_lock(
                        mem_a11_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=0,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    # dummy release
                    use_lock(
                        mem_a11_cons_prod_lock[row],
                        LockAction.Release,
                        value=0,
                    )
                    next_bd(block[15])
                with block[15]:
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    next_bd(block[16])
                with block[16]:
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=2 * a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    next_bd(block[17])
                with block[17]:
                    # dummy acquire
                    use_lock(
                        mem_a10_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=0,
                    )
                    dma_bd(
                        mem_a_buff_1[row],
                        offset=3 * a_out_step,
                        len=dimensions.a_length(),
                        dimensions=a_out_dimensions,
                    )
                    use_lock(
                        mem_a10_cons_prod_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[18])
                with block[18]:
                    # dummy acquire
                    use_lock(
                        mem_a11_cons_cons_lock[row],
                        LockAction.AcquireGreaterEqual,
                        value=0,
                    )
                    # dummy BD
                    dma_bd(
                        mem_a_buff_0[row],
                        offset=0,
                        len=0,
                    )
                    use_lock(
                        mem_a11_cons_prod_lock[row],
                        LockAction.Release,
                        value=1,
                    )
                    next_bd(block[7])
                with block[19]:
                    EndOp()

        for row in range(num_rows):
            make_a_memtile(row)

        # memtile b buffer layout:
        # M x (sample_blocks x T) x real/imaginary x (8 x 8)
        b_in_dimensions = [
            (
                dimensions.M,
                dimensions.num_sample_blocks * dimensions.b_length() // dimensions.M,
            ),
            (dimensions.simd_size(), dimensions.simd_size()),
            (2 * dimensions.T * dimensions.num_sample_blocks, dimensions.vmac_size()),
            (dimensions.simd_size(), 1),
        ]
        b_out_dimensions = [
            (
                dimensions.M,
                dimensions.num_sample_blocks * dimensions.b_length() // dimensions.M,
            ),
            # these two dimensions are not linearized because the max size of a dimension is 1023
            (dimensions.T, 2 * dimensions.vmac_size()),
            (2 * dimensions.vmac_size(), 1),
        ]
        b_out_step = 2 * dimensions.vmac_size() * dimensions.T

        def make_b_memtile(m):
            @memtile_dma(mem_tiles[2 * m + 1])
            def mem_body_mem_b(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(
                        mem_b_cons_prod_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=dimensions.num_sample_blocks,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.b_length(),
                        dimensions=b_in_dimensions,
                    )
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.Release,
                        value=dimensions.num_sample_blocks,
                    )
                    next_bd(block[2])
                with block[2]:
                    use_lock(
                        mem_b_cons_prod_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=dimensions.num_sample_blocks,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.b_length(),
                        dimensions=b_in_dimensions,
                    )
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.Release,
                        value=dimensions.num_sample_blocks,
                    )
                    next_bd(block[1])
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=dimensions.num_sample_blocks,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m + 1],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.b_length(),
                        dimensions=b_in_dimensions,
                    )
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.Release,
                        value=dimensions.num_sample_blocks,
                    )
                    next_bd(block[5])
                with block[5]:
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=dimensions.num_sample_blocks,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m + 1],
                        offset=0,
                        len=dimensions.num_sample_blocks * dimensions.b_length(),
                        dimensions=b_in_dimensions,
                    )
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.Release,
                        value=dimensions.num_sample_blocks,
                    )
                    next_bd(block[4])
                with block[6]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[15])
                with block[7]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m],
                        offset=0,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m],
                        offset=b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[9])
                with block[9]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m],
                        offset=2 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[10])
                with block[10]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m],
                        offset=3 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[11])
                with block[11]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m],
                        offset=0,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[12])
                with block[12]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m],
                        offset=b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[13])
                with block[13]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m],
                        offset=2 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[14]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m],
                        offset=3 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(mem_b_cons_prod_lock[2 * m], LockAction.Release, value=1)
                    next_bd(block[7])
                with block[15]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[16], chain=block[24])
                with block[16]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m + 1],
                        offset=0,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[17])
                with block[17]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m + 1],
                        offset=b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[18])
                with block[18]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m + 1],
                        offset=2 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[19])
                with block[19]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_0[2 * m + 1],
                        offset=3 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[20])
                with block[20]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m + 1],
                        offset=0,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[21])
                with block[21]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m + 1],
                        offset=b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[22])
                with block[22]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m + 1],
                        offset=2 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[23])
                with block[23]:
                    use_lock(
                        mem_b_cons_cons_lock[2 * m + 1],
                        LockAction.AcquireGreaterEqual,
                        value=1,
                    )
                    dma_bd(
                        mem_b_buff_1[2 * m + 1],
                        offset=3 * b_out_step,
                        len=dimensions.b_length(),
                        dimensions=b_out_dimensions,
                    )
                    use_lock(
                        mem_b_cons_prod_lock[2 * m + 1], LockAction.Release, value=1
                    )
                    next_bd(block[16])
                with block[24]:
                    EndOp()

        for m in range(num_cols // 2):
            make_b_memtile(m)

        acc_to_mem = [
            [
                object_fifo(
                    f"acc_to_mem_col{col}_row{row}",
                    compute_tiles[col][row],
                    mem_tiles[col],
                    1,
                    acc_ty,
                )
                for row in range(num_rows)
            ]
            for col in range(num_cols)
        ]
        acc_to_shim = [
            object_fifo(
                f"acc_to_shim_col{n}", mem_tiles[n], shim_tiles[n], 1, col_acc_ty
            )
            for n in range(num_cols)
        ]
        for n in range(num_cols):
            object_fifo_link(
                acc_to_mem[n],
                acc_to_shim[n],
                [row * dimensions.acc_length() for row in range(num_rows)],
            )

        x_engine_acc_zero = external_func(
            "x_engine_acc_zero",
            inputs=[acc_ty],
            link_with="build/x_engine.o",
        )
        x_engine_kernel = external_func(
            "x_engine_kernel",
            inputs=[a_ty, b_ty, acc_ty],
            link_with="build/x_engine.o",
        )

        def make_core(col, row):
            @core(compute_tiles[col][row])
            def core_body():
                for _ in range_(0x7FFFFFFF):
                    acc = acc_to_mem[col][row].acquire(ObjectFifoPort.Produce, 1)
                    x_engine_acc_zero(acc)
                    for _ in range_(
                        dimensions.num_sample_blocks * dimensions.integrations // 2
                    ):
                        for a, b in zip(
                            [a_buff_0[col][row], a_buff_1[col][row]],
                            [b_buff_0[col][row], b_buff_1[col][row]],
                        ):
                            use_lock(
                                a_cons_cons_lock[col][row],
                                LockAction.AcquireGreaterEqual,
                            )
                            use_lock(
                                b_cons_cons_lock[col][row],
                                LockAction.AcquireGreaterEqual,
                            )
                            x_engine_kernel(a, b, acc)
                            use_lock(a_cons_prod_lock[col][row], LockAction.Release)
                            use_lock(b_cons_prod_lock[col][row], LockAction.Release)
                    acc_to_mem[col][row].release(ObjectFifoPort.Produce, 1)

            @mem(compute_tiles[col][row])
            def mem_body(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(a_cons_prod_lock[col][row], LockAction.AcquireGreaterEqual)
                    dma_bd(a_buff_0[col][row])
                    use_lock(a_cons_cons_lock[col][row], LockAction.Release)
                    next_bd(block[2])
                with block[2]:
                    use_lock(a_cons_prod_lock[col][row], LockAction.AcquireGreaterEqual)
                    dma_bd(a_buff_1[col][row])
                    use_lock(a_cons_cons_lock[col][row], LockAction.Release)
                    next_bd(block[1])
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(b_cons_prod_lock[col][row], LockAction.AcquireGreaterEqual)
                    dma_bd(b_buff_0[col][row])
                    use_lock(b_cons_cons_lock[col][row], LockAction.Release)
                    next_bd(block[5])
                with block[5]:
                    use_lock(b_cons_prod_lock[col][row], LockAction.AcquireGreaterEqual)
                    dma_bd(b_buff_1[col][row])
                    use_lock(b_cons_cons_lock[col][row], LockAction.Release)
                    next_bd(block[4])
                with block[6]:
                    EndOp()

        for col in range(num_cols):
            for row in range(num_rows):
                make_core(col, row)

        tiles_to_trace = [
            compute_tiles[0][0],
            compute_tiles[0][0],
            mem_tiles[0],
            mem_tiles[1],
            shim_tiles[0],
            shim_tiles[1],
        ]
        if trace_size > 0:
            trace_utils.configure_trace(
                tiles_to_trace,
                coretile_events=[
                    CoreEvent.INSTR_EVENT_0,
                    CoreEvent.INSTR_EVENT_1,
                    CoreEvent.INSTR_VECTOR,
                    CoreEvent.MEMORY_STALL,
                    CoreEvent.LOCK_STALL,
                    PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
                    PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 1, True),
                ],
                coremem_events=[
                    MemEvent.DMA_S2MM_0_START_TASK,
                    MemEvent.DMA_MM2S_0_START_TASK,
                    MemEvent.CONFLICT_DM_BANK_0,
                    MemEvent.CONFLICT_DM_BANK_1,
                    MemEvent.CONFLICT_DM_BANK_2,
                    MemEvent.CONFLICT_DM_BANK_3,
                    MemEvent.EDGE_DETECTION_EVENT_0,
                    MemEvent.EDGE_DETECTION_EVENT_1,
                ],
                memtile_events=[
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True
                    ),  # DMA ch0 in
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_1, WireBundle.DMA, 1, True
                    ),  # DMA ch1 in
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_2, WireBundle.DMA, 2, True
                    ),  # DMA ch2 in
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_3, WireBundle.DMA, 3, True
                    ),  # DMA ch3 in
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_4, WireBundle.DMA, 0, False
                    ),  # DMA ch0 out
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_5, WireBundle.DMA, 1, False
                    ),  # DMA ch1 out
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_6, WireBundle.DMA, 2, False
                    ),  # DMA ch2 out
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_7, WireBundle.DMA, 3, False
                    ),  # DMA ch3 out
                ],
                shimtile_events=[
                    ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
                    ShimTileEvent.DMA_MM2S_0_STREAM_BACKPRESSURE,
                    ShimTileEvent.DMA_MM2S_0_STALLED_LOCK,
                    ShimTileEvent.DMA_MM2S_1_MEMORY_STARVATION,
                    ShimTileEvent.DMA_MM2S_1_STREAM_BACKPRESSURE,
                    ShimTileEvent.DMA_MM2S_1_STALLED_LOCK,
                ],
            )

        @runtime_sequence(input_ty, input_ty, all_acc_ty)
        def sequence(a, b, acc):
            if trace_size > 0:
                trace_utils.start_trace(trace_size=trace_size)

            # using size0 requires setting up repeat_count to size0 - 1
            a_tasks = [
                dma_configure_task(
                    shim_tiles[n],
                    DMAChannelDir.MM2S,
                    0,
                    repeat_count=dimensions.N_PFB - 1,
                )
                for n in range(num_cols)
            ]
            for n in range(num_cols):
                with bds(a_tasks[n]) as bd, bd[0]:
                    shim_dma_bd(
                        a,
                        offset=n
                        * dimensions.N
                        // 2
                        * dimensions.simd_size()
                        * dimensions.items_per_packet(),
                        sizes=[
                            dimensions.N_PFB,
                            dimensions.integrations,
                            dimensions.N // 2 * dimensions.simd_size(),
                            2 * dimensions.samples_per_packet(),
                        ],
                        strides=[
                            2 * dimensions.samples_per_packet(),
                            dimensions.items_per_packet() * dimensions.num_streams(),
                            dimensions.items_per_packet(),
                            1,
                        ],
                    )
                    EndOp()

            b_tasks = [
                dma_configure_task(
                    shim_tiles[n],
                    DMAChannelDir.MM2S,
                    1,
                    repeat_count=dimensions.N_PFB - 1,
                )
                for n in range(num_cols)
            ]
            for n in range(num_cols):
                with bds(b_tasks[n]) as bd, bd[0]:
                    shim_dma_bd(
                        b,
                        offset=n
                        * dimensions.M
                        * dimensions.simd_size()
                        * dimensions.items_per_packet(),
                        sizes=[
                            dimensions.N_PFB,
                            dimensions.integrations,
                            dimensions.M * dimensions.simd_size(),
                            2 * dimensions.samples_per_packet(),
                        ],
                        strides=[
                            2 * dimensions.samples_per_packet(),
                            dimensions.items_per_packet() * dimensions.num_streams(),
                            dimensions.items_per_packet(),
                            1,
                        ],
                    )
                    EndOp()

            assert dimensions.acc_length() % 512 == 0
            out_tasks = [
                shim_dma_single_bd_task(
                    acc_to_shim[n],
                    acc,
                    offset=n * dimensions.col_acc_length(),
                    # The last 2 dimensions are not linearized because the
                    # maximum size0 is 1023
                    sizes=[
                        1,
                        # make the run shorter if we are tracing
                        1 if trace_size else dimensions.N_PFB,
                        num_rows * dimensions.acc_length() // 512,
                        512,
                    ],
                    strides=[
                        0,
                        dimensions.all_acc_length() // dimensions.N_PFB,
                        512,
                        1,
                    ],
                    issue_token=True,
                )
                for n in range(num_cols)
            ]

            dma_start_task(*a_tasks)
            dma_start_task(*b_tasks)
            dma_start_task(*out_tasks)
            dma_await_task(*out_tasks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-size", default=0, type=int, help="Trace size")
    return parser.parse_args()


def main():
    args = parse_args()
    with mlir_mod_ctx() as ctx:
        x_engine(args.trace_size)
        res = ctx.module.operation.verify()
        if not res:
            raise RuntimeError(f"verify failed: {res}")
        print(ctx.module)


if __name__ == "__main__":
    main()
