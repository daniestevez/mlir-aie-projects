// x-engine
//
// X-engine correlator implemented on AIE-MLv2 using int8 multiplications.
//
// Copyright 2026 Daniel Estevez <daniel@destevez.net>
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <print>
#include <thread>
#include <vector>

#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

class BufferRing {
public:
  static constexpr uint32_t ring_size = 4; // this needs to be a power of 2
  std::array<xrt::bo, ring_size> a;
  std::array<xrt::bo, ring_size> b;
  std::atomic<std::uint32_t> available;

  BufferRing(xrt::device &device, xrt::kernel &kernel, size_t input_length) {
    for (unsigned int j = 0; j < ring_size; ++j) {
      a[j] = xrt::bo(device, input_length, XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(3));
      b[j] = xrt::bo(device, input_length, XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(4));
    }
    available = 0;
  }

  static constexpr uint32_t mask() { return ring_size - 1; }
};

class RingProducer {
public:
  std::shared_ptr<BufferRing> ring;
  uint32_t write_pointer = 0;

  RingProducer(std::shared_ptr<BufferRing> ring) : ring(ring) {}

  void produce() {
    // wait until ring is not full
    ring->available.wait(ring->ring_size, std::memory_order_acquire);

    // sync buffers to device
    ring->a[write_pointer & ring->mask()].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ring->b[write_pointer & ring->mask()].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    ring->available.fetch_add(1, std::memory_order_release);
    ring->available.notify_one();

    ++write_pointer;
  }
};

class RingConsumer {
public:
  std::shared_ptr<BufferRing> ring;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  xrt::bo bo_out;
  int instr_size;
  int read_pointer = 0;

  RingConsumer(std::shared_ptr<BufferRing> ring, xrt::device &device,
               xrt::kernel kernel, xrt::bo bo_instr, int all_acc_length,
               int instr_size)
      : ring(ring), kernel(kernel), bo_instr(bo_instr),
        bo_out(device, all_acc_length * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(5)),
        instr_size(instr_size) {}

  void consume() {
    // wait until ring is not empty
    ring->available.wait(0, std::memory_order_acquire);

    const unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_size,
                      ring->a[read_pointer & ring->mask()],
                      ring->b[read_pointer & ring->mask()], bo_out);
    run.wait();

    ring->available.fetch_sub(1, std::memory_order_release);
    ring->available.notify_one();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    ++read_pointer;
  }
};

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary("build/x_engine_insts.bin");

  xrt::device device;
  xrt::kernel kernel;
  const int verbosity = 0;
  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   "build/x_engine.xclbin", "MLIR_AIE");

  constexpr size_t input_length = 150405120;
  constexpr size_t all_acc_length = 1179648;
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto buffer_ring = std::make_shared<BufferRing>(device, kernel, input_length);
  auto producer = RingProducer(buffer_ring);
  auto consumer = RingConsumer(buffer_ring, device, kernel, bo_instr,
                               all_acc_length, instr_v.size());

  std::thread producer_thread([&producer]() {
    while (true) {
      producer.produce();
    }
    return;
  });

  auto t_measure = std::chrono::high_resolution_clock::now();
  uint64_t kernel_calls = 0;
  const double measurement_delta = 10.0;
  while (true) {
    consumer.consume();
    ++kernel_calls;
    const auto t_now = std::chrono::high_resolution_clock::now();
    const double delta =
        std::chrono::duration<double>(t_now - t_measure).count();
    if (delta > measurement_delta) {
      const double calls_per_second = static_cast<double>(kernel_calls) / delta;
      // This is calculated as
      // dimensions.samples_per_packet() * dimensions.integrations *
      // dimensions.N_PFB
      constexpr uint64_t samples_per_call = 293760;
      const double samples_per_second =
          static_cast<double>(samples_per_call) * calls_per_second;
      constexpr uint64_t num_streams = 256;
      constexpr uint64_t macs_per_call =
          4 * num_streams * num_streams * samples_per_call;
      const double tops =
          static_cast<double>(macs_per_call) * calls_per_second * 1e-12;
      std::println("{:.3f} Msps, {:.3f} TOPS, {:.3f} kernel calls/s",
                   samples_per_second * 1e-6, tops, calls_per_second);
      kernel_calls = 0;
      t_measure = t_now;
    }
  }

  return 0;
}
