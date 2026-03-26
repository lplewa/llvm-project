//===--- Level Zero V2 Dynamic Wrapper ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement wrapper for level_zero API calls through dlopen.
//
//===----------------------------------------------------------------------===//

#include <level_zero/ze_api.h>

#include "DLWrap.h"
#include "Shared/Debug.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm::offload::debug;

DLWRAP_INITIALIZE()

DLWRAP_INTERNAL(zeInit, 1)
DLWRAP(zeDriverGet, 2)
DLWRAP(zeDriverGetApiVersion, 2)
DLWRAP(zeContextCreate, 3)
DLWRAP(zeContextDestroy, 1)
DLWRAP(zeDeviceGet, 3)
DLWRAP(zeDeviceGetProperties, 2)
DLWRAP(zeDeviceGetComputeProperties, 2)
DLWRAP(zeDeviceGetMemoryProperties, 3)
DLWRAP(zeDeviceGetCacheProperties, 3)
DLWRAP(zeDeviceGetCommandQueueGroupProperties, 3)
DLWRAP(zeEventPoolCreate, 5)
DLWRAP(zeEventPoolDestroy, 1)
DLWRAP(zeEventCreate, 3)
DLWRAP(zeEventDestroy, 1)
DLWRAP(zeEventHostSynchronize, 2)
DLWRAP(zeEventHostReset, 1)
DLWRAP(zeEventQueryStatus, 1)
DLWRAP(zeCommandListCreateImmediate, 4)
DLWRAP(zeCommandListDestroy, 1)
DLWRAP(zeCommandListAppendSignalEvent, 2)
DLWRAP(zeCommandListAppendWaitOnEvents, 3)
DLWRAP(zeCommandListAppendBarrier, 4)
DLWRAP(zeModuleCreate, 5)
DLWRAP(zeModuleDestroy, 1)
DLWRAP(zeModuleGetNativeBinary, 3)
DLWRAP(zeModuleGetKernelNames, 3)
DLWRAP(zeModuleBuildLogDestroy, 1)
DLWRAP(zeModuleBuildLogGetString, 3)
DLWRAP(zeKernelCreate, 3)
DLWRAP(zeKernelDestroy, 1)
DLWRAP(zeKernelSetGroupSize, 4)
DLWRAP(zeKernelSuggestGroupSize, 7)
DLWRAP(zeKernelSetArgumentValue, 4)
DLWRAP(zeKernelGetProperties, 2)
DLWRAP(zeModuleGetGlobalPointer, 4)
DLWRAP(zeDriverGetExtensionFunctionAddress, 3)
DLWRAP(zeCommandListAppendLaunchKernel, 6)
DLWRAP(zeMemAllocDevice, 6)
DLWRAP(zeMemAllocHost, 5)
DLWRAP(zeMemAllocShared, 7)
DLWRAP(zeMemFree, 2)
DLWRAP(zeCommandListAppendMemoryCopy, 7)
DLWRAP(zeCommandListAppendMemoryFill, 8)
DLWRAP(zeCommandListHostSynchronize, 2)
DLWRAP(zeKernelSuggestMaxCooperativeGroupCount, 2)
DLWRAP(zeCommandListAppendLaunchCooperativeKernel, 6)

DLWRAP_FINALIZE()

#ifdef _WIN32
#define LEVEL_ZERO_LIBRARY "ze_loader.dll"
#else
#define LEVEL_ZERO_LIBRARY "libze_loader.so"
#endif

#ifndef TARGET_NAME
#error "Missing TARGET_NAME macro"
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME) " RTL"
#endif

static bool loadLevelZero() {
  std::string L0Library{LEVEL_ZERO_LIBRARY};
  std::string ErrMsg;

  ODBG(OLDT_Init) << "Trying to load " << L0Library;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library.c_str(),
                                                     &ErrMsg));

  constexpr uint32_t MinVersion{ZE_MAKE_VERSION(1, 10)};
  auto emitCheckVersion = [&]() {
    ODBG(OLDT_Init) << "Level Zero Loader compatible with version "
                    << ZE_MAJOR_VERSION(MinVersion) << "."
                    << ZE_MINOR_VERSION(MinVersion) << " is required";
  };

#ifndef _WIN32
  if (!DynlibHandle->isValid()) {
    L0Library +=
        std::string{"."} + std::to_string(ZE_MAJOR_VERSION(MinVersion));
    ErrMsg.clear();
    ODBG(OLDT_Init) << "Trying to load " << L0Library;
    DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
        llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library.c_str(),
                                                       &ErrMsg));
  }
#endif
  if (!DynlibHandle->isValid()) {
    if (ErrMsg.empty())
      ErrMsg = "unknown error";
    ODBG(OLDT_Init) << "Unable to load library '" << L0Library
                    << "': " << ErrMsg << "!";
    emitCheckVersion();
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      ODBG(OLDT_Init) << "Unable to find '" << Sym << "' in '" << L0Library
                      << "'!";
      emitCheckVersion();
      return false;
    }
    ODBG(OLDT_Init) << "Implementing " << Sym << " with dlsym(" << Sym
                    << ") -> " << P;

    *dlwrap::pointer(I) = P;
  }

  return true;
}

ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags) {
  if (!loadLevelZero())
    return ZE_RESULT_ERROR_UNKNOWN;
  return dlwrap_zeInit(flags);
}
