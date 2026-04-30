//===------- Offload API tests - olLaunchKernelWithPtrArgs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct LaunchKernelPtrArgsTestBase : OffloadQueueTest {
  void SetUpProgram(const char *program) {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary(program, Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));

    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {64, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};

    LaunchArgs.DynSharedMemory = 0;
  }

  void TearDown() override {
    if (Program) {
      olDestroyProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};

struct LaunchKernelPtrArgsSingleTestBase : LaunchKernelPtrArgsTestBase {
  void SetUpKernel(const char *kernel) {
    RETURN_ON_FATAL_FAILURE(SetUpProgram(kernel));
    ASSERT_SUCCESS(
        olGetSymbol(Program, kernel, OL_SYMBOL_KIND_KERNEL, &Kernel));
  }

  ol_symbol_handle_t Kernel = nullptr;
};

#define PTRARGS_KERNEL_TEST(NAME, KERNEL)                                      \
  struct olLaunchKernelWithPtrArgs##NAME##Test                                 \
      : LaunchKernelPtrArgsSingleTestBase {                                    \
    void SetUp() override { SetUpKernel(#KERNEL); }                            \
  };                                                                           \
  OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(                                    \
      olLaunchKernelWithPtrArgs##NAME##Test);

PTRARGS_KERNEL_TEST(Foo, foo)
PTRARGS_KERNEL_TEST(NoArgs, noargs)
PTRARGS_KERNEL_TEST(MultiArgs, multiargs)
PTRARGS_KERNEL_TEST(Composite, composite)
PTRARGS_KERNEL_TEST(Byte, byte)
PTRARGS_KERNEL_TEST(LocalMem, localmem)

struct LaunchKernelPtrArgsMultipleTestBase : LaunchKernelPtrArgsTestBase {
  void SetUpKernels(const char *program, std::vector<const char *> kernels) {
    RETURN_ON_FATAL_FAILURE(SetUpProgram(program));

    Kernels.resize(kernels.size());
    size_t I = 0;
    for (auto K : kernels)
      ASSERT_SUCCESS(
          olGetSymbol(Program, K, OL_SYMBOL_KIND_KERNEL, &Kernels[I++]));
  }

  std::vector<ol_symbol_handle_t> Kernels;
};

#define PTRARGS_KERNEL_MULTI_TEST(NAME, PROGRAM, ...)                          \
  struct olLaunchKernelWithPtrArgs##NAME##Test                                 \
      : LaunchKernelPtrArgsMultipleTestBase {                                  \
    void SetUp() override { SetUpKernels(#PROGRAM, {__VA_ARGS__}); }           \
  };                                                                           \
  OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(                                    \
      olLaunchKernelWithPtrArgs##NAME##Test);

PTRARGS_KERNEL_MULTI_TEST(Global, global, "write", "read")

TEST_P(olLaunchKernelWithPtrArgsFooTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  void *ArgPtrs[] = {&Mem};
  size_t ArgSizes[] = {sizeof(Mem)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelWithPtrArgsFooTest, SuccessThreaded) {
  threadify([&](size_t) {
    void *Mem;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                              LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

    void *ArgPtrs[] = {&Mem};
    size_t ArgSizes[] = {sizeof(Mem)};

    ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                             ArgSizes, &LaunchArgs));

    ASSERT_SUCCESS(olSyncQueue(Queue));

    uint32_t *Data = (uint32_t *)Mem;
    for (uint32_t i = 0; i < 64; i++) {
      ASSERT_EQ(Data[i], i);
    }

    ASSERT_SUCCESS(olMemFree(Mem));
  });
}

TEST_P(olLaunchKernelWithPtrArgsNoArgsTest, Success) {
  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, nullptr,
                                           nullptr, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olLaunchKernelWithPtrArgsMultiArgsTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(int), &Mem));

  char A = 3;
  int *B = (int *)Mem;
  short C = 5;

  void *ArgPtrs[] = {&A, &B, &C};
  size_t ArgSizes[] = {sizeof(A), sizeof(B), sizeof(C)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  int *Data = (int *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.GroupSize.x; i++)
    ASSERT_EQ(Data[i], A + C + static_cast<int>(i));

  ASSERT_SUCCESS(olMemFree(Mem));
}

struct Foo {
  int a;
  int b;
};

TEST_P(olLaunchKernelWithPtrArgsCompositeTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  uint32_t N = 1;
  Foo F{2, 3};
  uint32_t *Out = (uint32_t *)Mem;

  void *ArgPtrs[] = {&N, &F, &Out};
  size_t ArgSizes[] = {sizeof(N), sizeof(F), sizeof(Out)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.GroupSize.x; i++)
    ASSERT_EQ(Data[i], N + F.a + F.b + i);

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelWithPtrArgsFooTest, SuccessSynchronous) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  void *ArgPtrs[] = {&Mem};
  size_t ArgSizes[] = {sizeof(Mem)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(nullptr, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelWithPtrArgsByteTest, Success) {
  unsigned char C = 42;

  void *ArgPtrs[] = {&C};
  size_t ArgSizes[] = {sizeof(C)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olLaunchKernelWithPtrArgsLocalMemTest, Success) {
  LaunchArgs.NumGroups.x = 4;
  LaunchArgs.DynSharedMemory = 64 * sizeof(uint32_t);

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * LaunchArgs.NumGroups.x *
                                sizeof(uint32_t),
                            &Mem));

  void *ArgPtrs[] = {&Mem};
  size_t ArgSizes[] = {sizeof(Mem)};

  ASSERT_SUCCESS(olLaunchKernelWithPtrArgs(Queue, Device, Kernel, ArgPtrs,
                                           ArgSizes, &LaunchArgs));

  ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < LaunchArgs.GroupSize.x * LaunchArgs.NumGroups.x; i++)
    ASSERT_EQ(Data[i], (i % 64) * 2);

  ASSERT_SUCCESS(olMemFree(Mem));
}
