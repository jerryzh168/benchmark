#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

static void BM_MakeStorageImpl(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        c10::make_intrusive<at::StorageImpl>(
            options.dtype(),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }
}
BENCHMARK(BM_MakeStorageImpl);

static void BM_StorageImplCtor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  void* mem = malloc(sizeof(at::StorageImpl));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        new (mem) at::StorageImpl(
            options.dtype(),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }

  free(mem);
}
BENCHMARK(BM_StorageImplCtor);

static void BM_MallocStorageImpl(benchmark::State& state) {
  for (auto _ : state) {
    // NB: leaks memory
    benchmark::DoNotOptimize(malloc(sizeof(at::StorageImpl)));
  }
}
BENCHMARK(BM_MallocStorageImpl);

static void BM_TensorImplCtor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto storage = c10::make_intrusive<at::StorageImpl>(
      options.dtype(), 0, at::cuda::getCUDADeviceAllocator(), true);

  void* mem = malloc(sizeof(at::TensorImpl));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        new (mem) at::TensorImpl(storage, at::CUDATensorId(), false));
  }

  free(mem);
}
BENCHMARK(BM_TensorImplCtor);


static void BM_MallocTensorImpl(benchmark::State& state) {
  for (auto _ : state) {
    // NB: leaks memory
    benchmark::DoNotOptimize(malloc(sizeof(at::TensorImpl)));
  }
}
BENCHMARK(BM_MallocTensorImpl);

static void BM_Malloc_1(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(malloc(1));
  }
}
BENCHMARK(BM_Malloc_1);

static void BM_MakeTensorFromStorage(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto storage = c10::make_intrusive<at::StorageImpl>(
      options.dtype(), 0, at::cuda::getCUDADeviceAllocator(), true);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::detail::make_tensor<at::TensorImpl>(
          storage, at::CUDATensorId(), false));
  }
}
BENCHMARK(BM_MakeTensorFromStorage);

static void BM_MakeVariableFromTensor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        torch::autograd::make_variable(tmp, false));
  }
}
BENCHMARK(BM_MakeVariableFromTensor);

#define BENCHMARK_ALLOC(NAME, SIZE, DEVICE)                             \
  static void BM_ATenCPUTensorAllocation##NAME(benchmark::State& state) { \
    auto options = at::TensorOptions(DEVICE);                           \
    for (auto _ : state) {                                              \
      auto tmp = at::empty(SIZE, options);                              \
      benchmark::DoNotOptimize(tmp);                                    \
    }                                                                   \
  }                                                                     \
  BENCHMARK(BM_ATenCPUTensorAllocation##NAME);

#define BENCHMARK_CPU_ALLOC(NAME, SIZE) BENCHMARK_ALLOC(NAME, SIZE, at::kCPU)

BENCHMARK_CPU_ALLOC(Small1, {1})
BENCHMARK_CPU_ALLOC(Small2, {9})

BENCHMARK_CPU_ALLOC(Medium1, {32 * 32})
BENCHMARK_CPU_ALLOC(Medium2, {63 * 64})

BENCHMARK_CPU_ALLOC(Big1, {1024 * 1024})
BENCHMARK_CPU_ALLOC(Big2, {1024 * 8196})

BENCHMARK_MAIN();
