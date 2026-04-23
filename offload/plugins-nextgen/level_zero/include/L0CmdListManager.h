//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H

#include <mutex>

#include <level_zero/ze_api.h>

#include "L0Defs.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;

/// Owns one ze_command_list_handle_t and serializes appends on it.
/// L0 requires external synchronization for command list appends, and the
/// common plugin layer does not hold a lock across data transfer calls.
class L0CmdListManagerTy {
  L0DeviceTy &Device;
  ze_command_list_handle_t CmdList;
  std::mutex Mtx;

public:
  L0CmdListManagerTy(L0DeviceTy &Device, ze_command_list_handle_t CmdList)
      : Device(Device), CmdList(CmdList) {}
  ~L0CmdListManagerTy();

  L0CmdListManagerTy(const L0CmdListManagerTy &) = delete;
  L0CmdListManagerTy &operator=(const L0CmdListManagerTy &) = delete;
  L0CmdListManagerTy(L0CmdListManagerTy &&) = delete;
  L0CmdListManagerTy &operator=(L0CmdListManagerTy &&) = delete;

  Error appendKernelLaunch(ze_kernel_handle_t Kernel,
                           const ze_group_count_t &Groups,
                           ze_event_handle_t SignalEvent);
  Error appendMemCopy(void *Dst, const void *Src, size_t Size,
                      ze_event_handle_t SignalEvent);
  Error appendMemFill(void *Ptr, const void *Pattern, size_t PatternSize,
                      size_t Size, ze_event_handle_t SignalEvent);
  Error appendSignalEvent(ze_event_handle_t Event);
  Error appendWaitOnEvent(ze_event_handle_t Event);
  Error hostSynchronize(uint64_t TimeoutNs);
  Expected<bool> isComplete();
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H
