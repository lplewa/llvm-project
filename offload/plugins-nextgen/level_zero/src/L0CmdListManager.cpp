//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "L0CmdListManager.h"
#include "L0Device.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

L0CmdListManagerTy::~L0CmdListManagerTy() {
  if (CmdList)
    Device.releaseImmCmdList(CmdList);
}

Error L0CmdListManagerTy::appendKernelLaunch(ze_kernel_handle_t Kernel,
                                             const ze_group_count_t &Groups,
                                             ze_event_handle_t SignalEvent) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, Kernel, &Groups,
                    SignalEvent, 0, nullptr);
  return Plugin::success();
}

Error L0CmdListManagerTy::appendMemCopy(void *Dst, const void *Src, size_t Size,
                                        ze_event_handle_t SignalEvent) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                    SignalEvent, 0, nullptr);
  return Plugin::success();
}

Error L0CmdListManagerTy::appendMemFill(void *Ptr, const void *Pattern,
                                        size_t PatternSize, size_t Size,
                                        ze_event_handle_t SignalEvent) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, CmdList, Ptr, Pattern,
                    PatternSize, Size, SignalEvent, 0, nullptr);
  return Plugin::success();
}

Error L0CmdListManagerTy::appendSignalEvent(ze_event_handle_t Event) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CALL_ZE_RET_ERROR(zeCommandListAppendSignalEvent, CmdList, Event);
  return Plugin::success();
}

Error L0CmdListManagerTy::appendWaitOnEvent(ze_event_handle_t Event) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CALL_ZE_RET_ERROR(zeCommandListAppendWaitOnEvents, CmdList, 1, &Event);
  return Plugin::success();
}

// hostSynchronize and isComplete intentionally do not take Mtx: the L0 driver
// permits concurrent query against appends on the same immediate command list,
// and taking the append lock here would serialize a query behind every pending
// submission.
Error L0CmdListManagerTy::hostSynchronize(uint64_t TimeoutNs) {
  CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, CmdList, TimeoutNs);
  return Plugin::success();
}

Expected<bool> L0CmdListManagerTy::isComplete() {
  ze_result_t RC = zeCommandListHostSynchronize(CmdList, 0);
  if (RC == ZE_RESULT_SUCCESS)
    return true;
  if (RC == ZE_RESULT_NOT_READY)
    return false;
  return Plugin::error(ErrorCode::UNKNOWN,
                       "zeCommandListHostSynchronize query failed: 0x%x", RC);
}

} // namespace llvm::omp::target::plugin
