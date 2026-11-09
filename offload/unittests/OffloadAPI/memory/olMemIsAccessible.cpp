//===------- Offload API tests - olMemIsAccessible -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

using olMemIsAccessibleTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE(olMemIsAccessibleTest);

namespace {
// Returns a device other than Device that the fixture's Context was not
// created with, or nullptr.
ol_device_handle_t getForeignDevice(ol_device_handle_t Device) {
  ol_device_handle_t Host = TestEnvironment::getHostDevice();
  if (Host && Host != Device)
    return Host;
  for (auto D : TestEnvironment::getDevices())
    if (D.Handle != Device)
      return D.Handle;
  return nullptr;
}
} // namespace

// A range inside a device allocation is accessible from the device it was
// allocated on.
TEST_P(olMemIsAccessibleTest, SuccessDeviceAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A range inside a host allocation is accessible from the device.
TEST_P(olMemIsAccessibleTest, SuccessHostAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// Any range fully inside a liboffload allocation in Context is accessible.
TEST_P(olMemIsAccessibleTest, SuccessInteriorRange) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A range that ends exactly at the end of the allocation is accessible.
  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(
      Context, Device, static_cast<char *>(Ptr) + 512, 512, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A NULL Context is deprecated but still accepted.
TEST_P(olMemIsAccessibleTest, SuccessNoContext) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // Without a context the answer is backend specific, so only require that
  // the query itself succeeds.
  bool IsAccessible = false;
  ASSERT_SUCCESS(
      olMemIsAccessible(/*Context=*/nullptr, Device, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// For memory not allocated through liboffload the answer is backend specific
// and may be false.
TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleHostMemory) {
  // Plain host memory is not accessible from the device unless the system
  // provides unified shared memory.
  SKIP_KNOWN_FAILURE(AMDGPU("host memory is accessible when XNACK is enabled"));

  std::vector<char> HostBuffer(1024);

  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, HostBuffer.data(),
                                   HostBuffer.size(), &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  // The same pointer without a context must give the same answer.
  IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(/*Context=*/nullptr, Device,
                                   HostBuffer.data(), HostBuffer.size(),
                                   &IsAccessible));
  ASSERT_FALSE(IsAccessible);
}

// For a range not fully inside a liboffload allocation the answer is backend specific
// and may be false.
TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleBeyondAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A range extending past the end of the allocation is not accessible.
  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024 * 1024 * 1024,
                                   &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  // Same when starting from an interior pointer.
  IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device,
                                   static_cast<char *>(Ptr) + 512,
                                   1024 * 1024 * 1024, &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A range whose end overflows is not reported as accessible.
TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleSizeOverflow) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A size large enough to overflow `Ptr + Size` must not be reported as
  // accessible.
  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(
      Context, Device, Ptr, std::numeric_limits<size_t>::max(), &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// Ptr must not be NULL.
TEST_P(olMemIsAccessibleTest, InvalidNullPointer) {
  bool IsAccessible = false;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olMemIsAccessible(Context, Device, nullptr, 1024, &IsAccessible));
}

// IsAccessible must not be NULL.
TEST_P(olMemIsAccessibleTest, InvalidNullOutputPointer) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemIsAccessible(Context, Device, Ptr, 1024, nullptr));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// Device must not be NULL.
TEST_P(olMemIsAccessibleTest, InvalidNullDevice) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemIsAccessible(Context, nullptr, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A non-NULL Context requires Device to belong to it.
TEST_P(olMemIsAccessibleTest, InvalidDeviceNotInContext) {
  ol_device_handle_t Foreign = getForeignDevice(Device);
  if (!Foreign)
    GTEST_SKIP() << "No device outside the context available.";

  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE,
               olMemIsAccessible(Context, Foreign, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// IsAccessible is not written when the call fails.
TEST_P(olMemIsAccessibleTest, FailureLeavesOutputUntouched) {
  ol_device_handle_t Foreign = getForeignDevice(Device);
  if (!Foreign)
    GTEST_SKIP() << "No device outside the context available.";

  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  for (bool Sentinel : {false, true}) {
    bool IsAccessible = Sentinel;
    ASSERT_ERROR(OL_ERRC_INVALID_DEVICE,
                 olMemIsAccessible(Context, Foreign, Ptr, 1024, &IsAccessible));
    ASSERT_EQ(IsAccessible, Sentinel);

    IsAccessible = Sentinel;
    ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
                 olMemIsAccessible(Context, nullptr, Ptr, 1024, &IsAccessible));
    ASSERT_EQ(IsAccessible, Sentinel);

    IsAccessible = Sentinel;
    ASSERT_ERROR(
        OL_ERRC_INVALID_NULL_POINTER,
        olMemIsAccessible(Context, Device, nullptr, 1024, &IsAccessible));
    ASSERT_EQ(IsAccessible, Sentinel);
  }

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A range inside a managed allocation is accessible from the device.
TEST_P(olMemIsAccessibleTest, SuccessManagedAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// For a range not fully inside a liboffload allocation the answer is backend
// specific and does not depend on Context.
TEST_P(olMemIsAccessibleTest, PartialRangeAnsweredByBackend) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  void *Start = static_cast<char *>(Ptr) + 512;
  bool WithContext = false;
  bool WithoutContext = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Start, 1024, &WithContext));
  ASSERT_SUCCESS(olMemIsAccessible(/*Context=*/nullptr, Device, Start, 1024,
                                   &WithoutContext));
  ASSERT_EQ(WithContext, WithoutContext);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// Ptr need not stay valid after the call. This only checks that querying freed
// memory returns and leaves later queries working; its result is ignored.
TEST_P(olMemIsAccessibleTest, QueryAfterFree) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);
  ASSERT_SUCCESS(olMemFree(Context, Ptr));

  bool IsAccessible = false;
  (void)olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible);
  (void)olMemIsAccessible(/*Context=*/nullptr, Device, Ptr, 1024,
                          &IsAccessible);

  void *Live = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Live));
  ASSERT_NE(Live, nullptr);
  IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Live, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);
  ASSERT_SUCCESS(olMemFree(Context, Live));
}

// olMemIsAccessible is thread safe.
TEST_P(olMemIsAccessibleTest, ConcurrentQueries) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  threadify([&](size_t) {
    bool IsAccessible = false;
    ASSERT_SUCCESS(
        olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
    ASSERT_TRUE(IsAccessible);
  });

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

struct olMemIsAccessibleMultiDeviceTest : OffloadDeviceTest {};
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(
    olMemIsAccessibleMultiDeviceTest);

// Host and managed allocations are accessible from every device of the
// Context.
TEST_P(olMemIsAccessibleMultiDeviceTest, HostAndManagedAccessibleEverywhere) {
  // Allocate the managed pointer first so an unsupported skip leaks nothing.
  void *ManagedPtr = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olMemAlloc(
      Context, Devices[0], OL_ALLOC_TYPE_MANAGED, 1024, &ManagedPtr));
  ASSERT_NE(ManagedPtr, nullptr);
  void *HostPtr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(Context, Devices[0], 1024, &HostPtr));
  ASSERT_NE(HostPtr, nullptr);

  for (auto D : Devices) {
    bool IsAccessible = false;
    ASSERT_SUCCESS(olMemIsAccessible(Context, D, HostPtr, 1024, &IsAccessible));
    ASSERT_TRUE(IsAccessible);
    IsAccessible = false;
    ASSERT_SUCCESS(
        olMemIsAccessible(Context, D, ManagedPtr, 1024, &IsAccessible));
    ASSERT_TRUE(IsAccessible);
  }

  ASSERT_SUCCESS(olMemFree(Context, HostPtr));
  ASSERT_SUCCESS(olMemFree(Context, ManagedPtr));
}

// A device allocation is accessible from its owning device; the answer for
// other devices is backend specific.
TEST_P(olMemIsAccessibleMultiDeviceTest, DeviceAllocationOwnerAndPeer) {
  ASSERT_GE(Devices.size(), 2u);

  void *Ptr = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Devices[0], OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(
      olMemIsAccessible(Context, Devices[0], Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  // The peer answer is backend specific, so it must match the backend-only
  // query made with a NULL Context.
  bool PeerAccessible = false;
  ASSERT_SUCCESS(
      olMemIsAccessible(Context, Devices[1], Ptr, 1024, &PeerAccessible));
  bool BackendAccessible = false;
  ASSERT_SUCCESS(
      olMemIsAccessible(nullptr, Devices[1], Ptr, 1024, &BackendAccessible));
  ASSERT_EQ(PeerAccessible, BackendAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

// A Device outside a non-NULL Context is rejected.
TEST_P(olMemIsAccessibleMultiDeviceTest, InvalidDeviceOutsideContext) {
  ASSERT_GE(Devices.size(), 2u);

  ol_context_handle_t SingleContext = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Devices[0], &SingleContext));

  void *Ptr = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(SingleContext, Devices[0], OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_ERROR(
      OL_ERRC_INVALID_DEVICE,
      olMemIsAccessible(SingleContext, Devices[1], Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(SingleContext, Ptr));
  ASSERT_SUCCESS(olDestroyContext(SingleContext));
}
