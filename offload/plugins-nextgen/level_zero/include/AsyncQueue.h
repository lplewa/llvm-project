//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H

#include "L0Defs.h"
#include "L0Memory.h"

namespace llvm::omp::target::plugin {

/// Abstract queue backing a libomptarget __tgt_async_info.
struct L0AsyncQueueTy {
  virtual ~L0AsyncQueueTy() = default;

  virtual Error appendKernelLaunch(ze_kernel_handle_t Kernel,
                                   const ze_group_count_t &Groups,
                                   ze_event_handle_t SignalEvent) = 0;
  virtual Error appendKernelLaunchWithArguments(
      ze_kernel_handle_t Kernel, const ze_group_count_t &Groups,
      const ze_group_size_t &GroupSizes, void **ArgPtrs,
      ze_event_handle_t SignalEvent) = 0;
  virtual Error appendMemCopy(void *Dst, const void *Src, size_t Size,
                              ze_event_handle_t SignalEvent) = 0;
  virtual Error appendMemFill(void *Ptr, const void *Pattern,
                              size_t PatternSize, size_t Size,
                              ze_event_handle_t SignalEvent) = 0;
  virtual Error appendSignalEvent(ze_event_handle_t Event) = 0;
  virtual Error appendWaitOnEvent(ze_event_handle_t Event) = 0;
  virtual Error hostSynchronize(uint64_t TimeoutNs) = 0;
  virtual Expected<bool> isComplete() = 0;
  virtual Error destroy() = 0;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
