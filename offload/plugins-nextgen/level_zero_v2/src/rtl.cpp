//===--- Level Zero V2 Target RTL Implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for Level Zero V2 plugin.
//
//===----------------------------------------------------------------------===//

#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "Shared/Debug.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/TargetParser/Triple.h"

#include <level_zero/ze_api.h>

namespace llvm::omp::target::plugin {

using namespace error;
using namespace llvm::offload::debug;

/// Helper macro to call a Level Zero function and return a Plugin error on
/// failure.
#define CALL_ZE_RET_ERROR(Fn, ...)                                             \
  do {                                                                         \
    ze_result_t ZeResult = Fn(__VA_ARGS__);                                    \
    if (ZeResult != ZE_RESULT_SUCCESS)                                         \
      return Plugin::error(ErrorCode::UNKNOWN, "%s failed with error %d",      \
                           #Fn, static_cast<int>(ZeResult));                   \
  } while (0)

/// Helper macro to call a Level Zero function and store the result.
#define CALL_ZE(Rc, Fn, ...)                                                   \
  do {                                                                         \
    Rc = Fn(__VA_ARGS__);                                                      \
  } while (0)

/// Check if a Level Zero device is a GPU (following UR v2 adapter pattern:
/// only GPU devices are exposed).
static bool isDeviceGPU(ze_device_handle_t ZeDevice) {
  ze_device_properties_t Props{};
  Props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(ZeDevice, &Props) != ZE_RESULT_SUCCESS)
    return false;
  return Props.type == ZE_DEVICE_TYPE_GPU;
}

/// Global handler for the Level Zero V2 plugin.
/// Following UR v2 adapter: uses zeModuleGetGlobalPointer to resolve globals.
class L0V2GlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  // Defined out-of-line after L0V2DeviceImageTy.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override;
};

/// Kernel implementation for the Level Zero V2 plugin.
/// Following UR v2 adapter: kernel wraps a ze_kernel_handle_t created from
/// a ze_module_handle_t.
struct L0V2KernelTy final : public GenericKernelTy {
  ze_kernel_handle_t ZeKernel = nullptr;

  /// Number of kernel arguments from zeKernelGetProperties.
  uint32_t NumKernelArgs = 0;

  /// Per-argument sizes queried from the driver (via zexKernelGetArgumentSize
  /// extension or zeKernelGetProperties).
  std::unique_ptr<uint32_t[]> ArgSizes;

  L0V2KernelTy(const char *Name) : GenericKernelTy(Name) {}

  ~L0V2KernelTy() override {
    if (ZeKernel)
      zeKernelDestroy(ZeKernel);
  }

  // Defined out-of-line after L0V2DeviceImageTy.
  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override;

  // Defined out-of-line after L0V2DeviceTy.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], uint32_t DynBlockMemSize,
                   KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  Expected<uint64_t> maxGroupSize(GenericDeviceTy &GenericDevice,
                                  uint64_t DynamicMemSize) const override;

  // UR v2 adapter: zeKernelSuggestMaxCooperativeGroupCount.
  Expected<uint32_t>
  getMaxCooperativeGroupCount(GenericDeviceTy &GenericDevice,
                              const uint32_t NumThreads[3],
                              uint32_t DynBlockMemSize) const override;
};

// Forward declaration.
struct L0V2DeviceTy;

/// Device image that also holds the Level Zero module handle.
class L0V2DeviceImageTy : public DeviceImageTy {
  ze_module_handle_t ZeModule;

public:
  L0V2DeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                     std::unique_ptr<MemoryBuffer> &&MB,
                     ze_module_handle_t Module)
      : DeviceImageTy(ImageId, Device, std::move(MB)), ZeModule(Module) {}

  ze_module_handle_t getModule() const { return ZeModule; }
};

/// Device implementation for the Level Zero V2 plugin.
struct L0V2DeviceTy final : public GenericDeviceTy {
  /// Level Zero device handle.
  ze_device_handle_t ZeDevice;

  /// Level Zero context shared by all devices under the same driver.
  ze_context_handle_t ZeContext;

  /// Level Zero driver handle (for API version queries).
  ze_driver_handle_t ZeDriver;

  /// Device properties queried from Level Zero.
  ze_device_properties_t ZeDeviceProperties{};
  ze_device_compute_properties_t ZeDeviceComputeProperties{};
  /// All memory modules on the device (UR v2 pattern: query all, not just one).
  llvm::SmallVector<ze_device_memory_properties_t, 2> ZeDeviceMemoryProperties;
  ze_device_cache_properties_t ZeDeviceCacheProperties{};

  /// Command queue group ordinal for compute.
  uint32_t ComputeQueueGroupOrdinal = std::numeric_limits<uint32_t>::max();
  uint32_t ComputeQueueGroupCount = 0;

  /// Command queue group ordinal for copy.
  uint32_t CopyQueueGroupOrdinal = std::numeric_limits<uint32_t>::max();
  uint32_t CopyQueueGroupCount = 0;

  /// Driver version string from zeDriverGetApiVersion.
  std::string DriverVersionStr;

  /// Cached global memory size (sum of all memory modules).
  uint64_t GlobalMemSize = 0;

  /// Whether the device supports cooperative kernels.
  bool SupportsCooperativeKernels = false;

  /// Extension: zeCommandListAppendHostFunction (queried via
  /// zeDriverGetExtensionFunctionAddress).
  /// Signature from zex_cmdlist.h in the UR v2 adapter.
  using ZeCommandListAppendHostFunctionT = ze_result_t(ZE_APICALL *)(
      ze_command_list_handle_t, void *, void *, void *,
      ze_event_handle_t, uint32_t, ze_event_handle_t *);
  ZeCommandListAppendHostFunctionT ZeCommandListAppendHostFunction = nullptr;

  /// Event pool for this device (UR v2 pattern: host-visible events).
  ze_event_pool_handle_t ZeEventPool = nullptr;
  static constexpr uint32_t EventPoolSize = 64;
  uint32_t NextEventIndex = 0;

  L0V2DeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
               ze_device_handle_t ZeDevice, ze_context_handle_t ZeContext,
               ze_driver_handle_t ZeDriver)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, SPIRVGridValues),
        ZeDevice(ZeDevice), ZeContext(ZeContext), ZeDriver(ZeDriver) {
    ZeDeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZeDeviceProperties.pNext = nullptr;
    ZeDeviceComputeProperties.stype =
        ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    ZeDeviceComputeProperties.pNext = nullptr;
    ZeDeviceCacheProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    ZeDeviceCacheProperties.pNext = nullptr;
  }

  // Memory allocation using clean Level Zero API (not UMF).
  Expected<void *> allocate(size_t Size, void *HstPtr,
                            TargetAllocTy Kind) override {
    void *Ptr = nullptr;
    // UR v2 adapter passes 0 (driver chooses suitable alignment).
    constexpr size_t Alignment = 0;

    switch (Kind) {
    case TARGET_ALLOC_DEVICE: {
      ze_device_mem_alloc_desc_t DevDesc{};
      DevDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      CALL_ZE_RET_ERROR(zeMemAllocDevice, ZeContext, &DevDesc, Size, Alignment,
                        ZeDevice, &Ptr);
      break;
    }
    case TARGET_ALLOC_HOST: {
      ze_host_mem_alloc_desc_t HostDesc{};
      HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
      CALL_ZE_RET_ERROR(zeMemAllocHost, ZeContext, &HostDesc, Size, Alignment,
                        &Ptr);
      break;
    }
    case TARGET_ALLOC_SHARED:
    default: {
      ze_device_mem_alloc_desc_t DevDesc{};
      DevDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      ze_host_mem_alloc_desc_t HostDesc{};
      HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
      CALL_ZE_RET_ERROR(zeMemAllocShared, ZeContext, &DevDesc, &HostDesc, Size,
                        Alignment, ZeDevice, &Ptr);
      break;
    }
    }
    return Ptr;
  }

  Error free(void *TgtPtr, TargetAllocTy Kind) override {
    CALL_ZE_RET_ERROR(zeMemFree, ZeContext, TgtPtr);
    return Plugin::success();
  }

  Error setContext() override { return Plugin::success(); }

  Error initImpl(GenericPluginTy &Plugin) override {
    // Query device properties (following UR v2 adapter pattern).
    CALL_ZE_RET_ERROR(zeDeviceGetProperties, ZeDevice, &ZeDeviceProperties);
    CALL_ZE_RET_ERROR(zeDeviceGetComputeProperties, ZeDevice,
                      &ZeDeviceComputeProperties);

    // Query ALL memory modules, not just the first one.
    // UR v2 adapter enumerates all memory modules for accurate totals.
    uint32_t MemCount = 0;
    CALL_ZE_RET_ERROR(zeDeviceGetMemoryProperties, ZeDevice, &MemCount,
                      nullptr);
    if (MemCount > 0) {
      ZeDeviceMemoryProperties.resize(MemCount);
      for (auto &P : ZeDeviceMemoryProperties) {
        P.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        P.pNext = nullptr;
      }
      CALL_ZE_RET_ERROR(zeDeviceGetMemoryProperties, ZeDevice, &MemCount,
                        ZeDeviceMemoryProperties.data());
    }

    // Cache properties.
    uint32_t CacheCount = 1;
    CALL_ZE_RET_ERROR(zeDeviceGetCacheProperties, ZeDevice, &CacheCount,
                      &ZeDeviceCacheProperties);

    ODBG(OLDT_Device) << "Found a GPU device, Name = "
                      << ZeDeviceProperties.name;

    // Build device UID from UUID.
    std::string Uid;
    for (int N = 0; N < ZE_MAX_DEVICE_UUID_SIZE; N++)
      Uid += std::to_string(ZeDeviceProperties.uuid.id[N]);
    setDeviceUidFromVendorUid(Uid);

    // Query driver API version.
    ze_api_version_t APIVersion = ZE_API_VERSION_CURRENT;
    CALL_ZE_RET_ERROR(zeDriverGetApiVersion, ZeDriver, &APIVersion);
    DriverVersionStr = std::to_string(ZE_MAJOR_VERSION(APIVersion)) + "." +
                       std::to_string(ZE_MINOR_VERSION(APIVersion));

    // Calculate global memory size by summing all memory modules
    // (following UR v2 adapter: calculateGlobalMemSize).
    GlobalMemSize = 0;
    for (const auto &MemProp : ZeDeviceMemoryProperties)
      GlobalMemSize += MemProp.totalSize;

    // Find compute and copy queue group ordinals.
    if (auto Err = findQueueGroupOrdinals())
      return Err;

    // Cooperative support detected during findQueueGroupOrdinals.

    // Query zeCommandListAppendHostFunction extension.
    ze_result_t ExtRC = zeDriverGetExtensionFunctionAddress(
        ZeDriver, "zeCommandListAppendHostFunction",
        reinterpret_cast<void **>(&ZeCommandListAppendHostFunction));
    if (ExtRC != ZE_RESULT_SUCCESS)
      ZeCommandListAppendHostFunction = nullptr;

    // Create event pool (UR v2 adapter pattern: host-visible events).
    ze_event_pool_desc_t PoolDesc{};
    PoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    PoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    PoolDesc.count = EventPoolSize;
    CALL_ZE_RET_ERROR(zeEventPoolCreate, ZeContext, &PoolDesc, 1, &ZeDevice,
                      &ZeEventPool);

    return Plugin::success();
  }

  Error deinitImpl() override {
    if (ZeEventPool)
      CALL_ZE_RET_ERROR(zeEventPoolDestroy, ZeEventPool);
    ZeEventPool = nullptr;
    return Plugin::success();
  }

  Expected<DeviceImageTy *>
  loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                 int32_t ImageId) override {
    // Determine the module format from the input image.
    // Following UR v2 adapter: detect SPIR-V vs native format.
    StringRef Data(TgtImage->getBufferStart(), TgtImage->getBufferSize());
    ze_module_format_t ModuleFormat = ZE_MODULE_FORMAT_IL_SPIRV;

    // Check if this is an OffloadBinary wrapper (contains inner SPIR-V).
    if (identify_magic(Data) == file_magic::offload_binary) {
      auto BinsOrErr = object::OffloadBinary::create(
          MemoryBufferRef(Data, "offload_binary"));
      if (!BinsOrErr)
        return BinsOrErr.takeError();
      if (BinsOrErr->size() == 1) {
        auto *Inner = (*BinsOrErr)[0].get();
        Data = Inner->getImage();
        if (Inner->getImageKind() == object::IMG_SPIRV)
          ModuleFormat = ZE_MODULE_FORMAT_IL_SPIRV;
        else
          ModuleFormat = ZE_MODULE_FORMAT_NATIVE;
      }
    } else if (identify_magic(Data) == file_magic::spirv_object) {
      ModuleFormat = ZE_MODULE_FORMAT_IL_SPIRV;
    } else {
      // Assume native binary (ELF from driver).
      ModuleFormat = ZE_MODULE_FORMAT_NATIVE;
    }

    // Create Level Zero module (following UR v2 adapter: zeModuleCreate).
    ze_module_desc_t ModuleDesc{};
    ModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    ModuleDesc.format = ModuleFormat;
    ModuleDesc.inputSize = Data.size();
    ModuleDesc.pInputModule =
        reinterpret_cast<const uint8_t *>(Data.data());
    ModuleDesc.pBuildFlags = "";
    ModuleDesc.pConstants = nullptr;

    ze_module_handle_t ZeModule = nullptr;
    ze_module_build_log_handle_t BuildLog = nullptr;
    CALL_ZE_RET_ERROR(zeModuleCreate, ZeContext, ZeDevice, &ModuleDesc,
                      &ZeModule, &BuildLog);
    if (BuildLog)
      zeModuleBuildLogDestroy(BuildLog);

    // Extract native binary (ELF) from the driver so the offload framework
    // can work with it. This is the key workaround: the framework expects
    // ELF, but we compiled from SPIR-V. zeModuleGetNativeBinary returns
    // the driver-compiled ELF.
    size_t NativeBinSize = 0;
    CALL_ZE_RET_ERROR(zeModuleGetNativeBinary, ZeModule, &NativeBinSize,
                      nullptr);
    std::vector<uint8_t> NativeBin(NativeBinSize);
    CALL_ZE_RET_ERROR(zeModuleGetNativeBinary, ZeModule, &NativeBinSize,
                      NativeBin.data());

    auto NativeBuffer = MemoryBuffer::getMemBufferCopy(
        StringRef(reinterpret_cast<const char *>(NativeBin.data()),
                  NativeBinSize),
        "L0V2 Native Binary");

    // Allocate image with the native ELF and store the module handle.
    auto *Image = Plugin.allocate<L0V2DeviceImageTy>();
    new (Image) L0V2DeviceImageTy(ImageId, *this, std::move(NativeBuffer),
                                   ZeModule);
    return Image;
  }

  Error unloadBinaryImpl(DeviceImageTy *Image) override {
    auto *L0Image = static_cast<L0V2DeviceImageTy *>(Image);
    ze_module_handle_t ZeModule = L0Image->getModule();
    if (ZeModule)
      CALL_ZE_RET_ERROR(zeModuleDestroy, ZeModule);
    return Plugin::success();
  }

  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override {
    // UR v2 adapter pattern: zeCommandListHostSynchronize on the immediate
    // command list. This waits for all previously appended commands to complete.
    auto *ImmCmdList =
        static_cast<ze_command_list_handle_t>(AsyncInfo.Queue);
    if (!ImmCmdList)
      return Plugin::success();

    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList, UINT64_MAX);

    return Plugin::success();
  }

  Error queryAsyncImpl(__tgt_async_info &AsyncInfo, bool ReleaseQueue,
                       bool *IsQueueWorkCompleted) override {
    // UR v2 adapter pattern: non-blocking query via timeout=0.
    auto *ImmCmdList =
        static_cast<ze_command_list_handle_t>(AsyncInfo.Queue);
    if (!ImmCmdList) {
      if (IsQueueWorkCompleted)
        *IsQueueWorkCompleted = true;
      return Plugin::success();
    }

    ze_result_t Status = zeCommandListHostSynchronize(ImmCmdList, 0);
    if (IsQueueWorkCompleted)
      *IsQueueWorkCompleted = (Status == ZE_RESULT_SUCCESS);
    return Plugin::success();
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    // On integrated devices, host pointers are directly accessible.
    return HstPtr;
  }

  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    return false;
  }

  // Data transfer using Level Zero immediate command lists.

  /// Get or create the immediate command list from the async info wrapper.
  Expected<ze_command_list_handle_t>
  getOrCreateImmCmdList(AsyncInfoWrapperTy &AsyncInfoWrapper) {
    auto ImmCmdList = AsyncInfoWrapper.getQueueAs<ze_command_list_handle_t>();
    if (!ImmCmdList) {
      ze_command_queue_desc_t QueueDesc{};
      QueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
      QueueDesc.ordinal = ComputeQueueGroupOrdinal;
      QueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
      CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, ZeContext, ZeDevice,
                        &QueueDesc, &ImmCmdList);
      AsyncInfoWrapper.setQueueAs<ze_command_list_handle_t>(ImmCmdList);
    }
    return ImmCmdList;
  }

  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto ImmCmdListOrErr = getOrCreateImmCmdList(AsyncInfoWrapper);
    if (!ImmCmdListOrErr)
      return ImmCmdListOrErr.takeError();
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, *ImmCmdListOrErr, TgtPtr,
                      HstPtr, Size, nullptr, 0, nullptr);
    return Plugin::success();
  }

  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto ImmCmdListOrErr = getOrCreateImmCmdList(AsyncInfoWrapper);
    if (!ImmCmdListOrErr)
      return ImmCmdListOrErr.takeError();
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, *ImmCmdListOrErr, HstPtr,
                      TgtPtr, Size, nullptr, 0, nullptr);
    return Plugin::success();
  }

  Error dataFence(__tgt_async_info *AsyncInfo) override {
    return Plugin::success();
  }

  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto ImmCmdListOrErr = getOrCreateImmCmdList(AsyncInfoWrapper);
    if (!ImmCmdListOrErr)
      return ImmCmdListOrErr.takeError();
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, *ImmCmdListOrErr, DstPtr,
                      SrcPtr, Size, nullptr, 0, nullptr);
    return Plugin::success();
  }

  Error dataFillImpl(void *TgtPtr, const void *PatternPtr,
                     int64_t PatternSize, int64_t Size,
                     AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto ImmCmdListOrErr = getOrCreateImmCmdList(AsyncInfoWrapper);
    if (!ImmCmdListOrErr)
      return ImmCmdListOrErr.takeError();
    auto ImmCmdList = *ImmCmdListOrErr;

    // UR v2 adapter pattern: zeCommandListAppendMemoryFill requires
    // patternSize to be a power of 2. When it's not, emulate with
    // repeated zeCommandListAppendMemoryCopy.
    bool IsPowerOf2 = PatternSize > 0 && (PatternSize & (PatternSize - 1)) == 0;
    if (IsPowerOf2) {
      CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, ImmCmdList, TgtPtr,
                        PatternPtr, PatternSize, Size, nullptr, 0, nullptr);
    } else {
      uint64_t NumCopies = Size / PatternSize;
      for (uint64_t I = 0; I < NumCopies; I++) {
        void *Dst = static_cast<uint8_t *>(TgtPtr) + I * PatternSize;
        CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, ImmCmdList, Dst,
                          PatternPtr, PatternSize, nullptr, 0, nullptr);
      }
    }
    return Plugin::success();
  }

  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // UR v2 adapter pattern: create an immediate command list for this queue.
    ze_command_queue_desc_t QueueDesc{};
    QueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    QueueDesc.ordinal = ComputeQueueGroupOrdinal;
    QueueDesc.index = 0;
    QueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

    ze_command_list_handle_t ImmCmdList = nullptr;
    CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, ZeContext, ZeDevice,
                      &QueueDesc, &ImmCmdList);

    AsyncInfoWrapper.setQueueAs<ze_command_list_handle_t>(ImmCmdList);
    return Plugin::success();
  }

  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &AsyncInfo) override {
    auto ImmCmdList = AsyncInfo.getQueueAs<ze_command_list_handle_t>();

    // Use zeCommandListAppendHostFunction extension if available
    // (following UR v2 adapter: appendHostTaskExp pattern).
    if (ZeCommandListAppendHostFunction && ImmCmdList) {
      CALL_ZE_RET_ERROR(ZeCommandListAppendHostFunction, ImmCmdList,
                        reinterpret_cast<void *>(Callback), UserData,
                        nullptr, nullptr, 0, nullptr);
      return Plugin::success();
    }

    // Fallback: synchronize then run on host.
    if (ImmCmdList)
      CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList, UINT64_MAX);

    Callback(UserData);
    return Plugin::success();
  }

  // Event implementation following UR v2 adapter pattern:
  // Events are ze_event_handle_t allocated from the device's event pool.

  Error createEventImpl(void **EventPtrStorage) override {
    ze_event_desc_t EvDesc{};
    EvDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    EvDesc.index = NextEventIndex % EventPoolSize;
    NextEventIndex++;
    EvDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    EvDesc.wait = 0;

    ze_event_handle_t ZeEvent = nullptr;
    CALL_ZE_RET_ERROR(zeEventCreate, ZeEventPool, &EvDesc, &ZeEvent);
    *EventPtrStorage = ZeEvent;
    return Plugin::success();
  }

  Error destroyEventImpl(void *EventPtr) override {
    auto ZeEvent = static_cast<ze_event_handle_t>(EventPtr);
    CALL_ZE_RET_ERROR(zeEventDestroy, ZeEvent);
    return Plugin::success();
  }

  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // UR v2 adapter pattern: append signal event to the immediate command list.
    auto ZeEvent = static_cast<ze_event_handle_t>(EventPtr);
    auto ImmCmdList = AsyncInfoWrapper.getQueueAs<ze_command_list_handle_t>();
    if (!ImmCmdList)
      return Plugin::success();

    CALL_ZE_RET_ERROR(zeCommandListAppendSignalEvent, ImmCmdList, ZeEvent);
    return Plugin::success();
  }

  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    auto ZeEvent = static_cast<ze_event_handle_t>(EventPtr);
    auto ImmCmdList = AsyncInfoWrapper.getQueueAs<ze_command_list_handle_t>();
    if (!ImmCmdList)
      return Plugin::success();

    CALL_ZE_RET_ERROR(zeCommandListAppendWaitOnEvents, ImmCmdList, 1, &ZeEvent);
    return Plugin::success();
  }

  Expected<bool>
  isEventCompleteImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // UR v2 adapter pattern: zeEventQueryStatus.
    auto ZeEvent = static_cast<ze_event_handle_t>(EventPtr);
    ze_result_t Status = zeEventQueryStatus(ZeEvent);
    return (Status == ZE_RESULT_SUCCESS);
  }

  Error syncEventImpl(void *EventPtr) override {
    // UR v2 adapter pattern: zeEventHostSynchronize with infinite timeout.
    auto ZeEvent = static_cast<ze_event_handle_t>(EventPtr);
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, ZeEvent, UINT64_MAX);
    return Plugin::success();
  }

  Expected<InfoTreeNode> obtainInfoImpl() override {
    InfoTreeNode Info;

    // Device name from Level Zero properties.
    Info.add("Device Name", std::string(ZeDeviceProperties.name), "",
             DeviceInfo::NAME);
    Info.add("Product Name", std::string(ZeDeviceProperties.name), "",
             DeviceInfo::PRODUCT_NAME);

    // UID from device UUID (hex-formatted).
    std::string UidStr;
    for (int N = 0; N < ZE_MAX_DEVICE_UUID_SIZE; N++) {
      char Buf[4];
      snprintf(Buf, sizeof(Buf), "%02x", ZeDeviceProperties.uuid.id[N]);
      if (N > 0)
        UidStr += ":";
      UidStr += Buf;
    }
    Info.add("Device UID", UidStr, "", DeviceInfo::UID);

    Info.add("Device Type", "GPU", "", DeviceInfo::TYPE);
    // UR v2 adapter: "Level-Zero does not return vendor's name at the moment"
    // but we know it's Intel for Level Zero devices.
    Info.add("Vendor", "Intel", "", DeviceInfo::VENDOR);
    Info.add("Vendor ID", static_cast<uint64_t>(ZeDeviceProperties.vendorId),
             "", DeviceInfo::VENDOR_ID);
    Info.add("Driver Version", DriverVersionStr, "", DeviceInfo::DRIVER_VERSION);

    // UR v2 adapter: numEUsPerSubslice * numSubslicesPerSlice * numSlices
    uint64_t MaxComputeUnits =
        static_cast<uint64_t>(ZeDeviceProperties.numEUsPerSubslice) *
        ZeDeviceProperties.numSubslicesPerSlice * ZeDeviceProperties.numSlices;
    Info.add("Number of compute units", MaxComputeUnits, "",
             DeviceInfo::NUM_COMPUTE_UNITS);

    // UR v2 adapter: maxTotalGroupSize from compute properties.
    Info.add(
        "Max Work Group Size",
        static_cast<uint64_t>(ZeDeviceComputeProperties.maxTotalGroupSize), "",
        DeviceInfo::MAX_WORK_GROUP_SIZE);

    // UR v2 adapter: per-dimension max group sizes.
    auto &WGSizeDim =
        *Info.add("Workgroup Max Size per Dimension", std::monostate{}, "",
                  DeviceInfo::MAX_WORK_GROUP_SIZE_PER_DIMENSION);
    WGSizeDim.add(
        "x", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeX));
    WGSizeDim.add(
        "y", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeY));
    WGSizeDim.add(
        "z", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeZ));

    // Max total work items, capped to uint32_t (queried as uint32_t by
    // liboffload).
    uint64_t MaxWorkSize =
        static_cast<uint64_t>(ZeDeviceComputeProperties.maxTotalGroupSize) *
        ZeDeviceComputeProperties.maxGroupCountX;
    MaxWorkSize =
        std::min(MaxWorkSize,
                 static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    Info.add("Maximum Grid Dimensions", MaxWorkSize, "",
             DeviceInfo::MAX_WORK_SIZE);

    // UR v2 adapter: per-dimension max group counts * max group sizes.
    auto &GridDim = *Info.add("Grid Size per Dimension", std::monostate{}, "",
                              DeviceInfo::MAX_WORK_SIZE_PER_DIMENSION);
    GridDim.add(
        "x", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeX) *
                 ZeDeviceComputeProperties.maxGroupCountX);
    GridDim.add(
        "y", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeY) *
                 ZeDeviceComputeProperties.maxGroupCountY);
    GridDim.add(
        "z", static_cast<uint64_t>(ZeDeviceComputeProperties.maxGroupSizeZ) *
                 ZeDeviceComputeProperties.maxGroupCountZ);

    // UR v2 adapter: maxSharedLocalMemory from compute properties.
    Info.add(
        "Local memory size (bytes)",
        static_cast<uint64_t>(ZeDeviceComputeProperties.maxSharedLocalMemory),
        "", DeviceInfo::WORK_GROUP_LOCAL_MEM_SIZE);

    Info.add("Global memory size (bytes)", GlobalMemSize, "",
             DeviceInfo::GLOBAL_MEM_SIZE);
    Info.add("Max Memory Allocation Size (bytes)",
             static_cast<uint64_t>(ZeDeviceProperties.maxMemAllocSize), "",
             DeviceInfo::MAX_MEM_ALLOC_SIZE);

    // UR v2 adapter: coreClockRate from device properties.
    Info.add("Max clock frequency (MHz)",
             static_cast<uint64_t>(ZeDeviceProperties.coreClockRate), "",
             DeviceInfo::MAX_CLOCK_FREQUENCY);

    // UR v2 adapter: minimum maxClockRate across all memory modules.
    // If there are no memory modules, report 0.
    uint32_t MemClockRate = 0;
    if (!ZeDeviceMemoryProperties.empty()) {
      MemClockRate = ZeDeviceMemoryProperties[0].maxClockRate;
      for (const auto &MemProp : ZeDeviceMemoryProperties)
        MemClockRate = std::min(MemClockRate, MemProp.maxClockRate);
    }
    Info.add("Max memory clock frequency (MHz)",
             static_cast<uint64_t>(MemClockRate), "",
             DeviceInfo::MEMORY_CLOCK_RATE);

    // UR v2 adapter: hard-coded as 64.
    Info.add("Memory Address Size", uint64_t{64u}, "bits",
             DeviceInfo::ADDRESS_BITS);

    Info.add("Cooperative launch support", SupportsCooperativeKernels, "",
             DeviceInfo::COOPERATIVE_LAUNCH_SUPPORT);

    return Info;
  }

  Expected<bool>
  hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return true;
  }

  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    auto *Kernel = Plugin.allocate<L0V2KernelTy>();
    if (!Kernel)
      return Plugin::error(ErrorCode::UNKNOWN,
                           "Failed to allocate memory for kernel");
    new (Kernel) L0V2KernelTy(Name);
    return *Kernel;
  }

  Error getDeviceMemorySize(uint64_t &DSize) override {
    DSize = GlobalMemSize;
    return Plugin::success();
  }

  Error getDeviceStackSize(uint64_t &V) override {
    V = 0;
    return Plugin::success();
  }

  Error setDeviceStackSize(uint64_t V) override { return Plugin::success(); }

private:
  /// Discover compute and copy command queue group ordinals.
  Error findQueueGroupOrdinals() {
    uint32_t NumGroups = 0;
    CALL_ZE_RET_ERROR(zeDeviceGetCommandQueueGroupProperties, ZeDevice,
                      &NumGroups, nullptr);

    llvm::SmallVector<ze_command_queue_group_properties_t> GroupProps(NumGroups);
    for (auto &P : GroupProps) {
      P.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
      P.pNext = nullptr;
    }
    CALL_ZE_RET_ERROR(zeDeviceGetCommandQueueGroupProperties, ZeDevice,
                      &NumGroups, GroupProps.data());

    for (uint32_t I = 0; I < NumGroups; I++) {
      if (GroupProps[I].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        ComputeQueueGroupOrdinal = I;
        ComputeQueueGroupCount = GroupProps[I].numQueues;
      }
      // Prefer a dedicated copy engine (no compute flag).
      if ((GroupProps[I].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
          !(GroupProps[I].flags &
            ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
        CopyQueueGroupOrdinal = I;
        CopyQueueGroupCount = GroupProps[I].numQueues;
      }
      // Check cooperative kernel support (from L0 plugin:
      // checkCooperativeKernelSupport).
      if (GroupProps[I].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS)
        SupportsCooperativeKernels = true;
    }

    ODBG(OLDT_Device) << "Compute ordinal: " << ComputeQueueGroupOrdinal
                      << " (" << ComputeQueueGroupCount << " queues)";
    if (CopyQueueGroupOrdinal != std::numeric_limits<uint32_t>::max())
      ODBG(OLDT_Device) << "Copy ordinal: " << CopyQueueGroupOrdinal << " ("
                        << CopyQueueGroupCount << " queues)";

    return Plugin::success();
  }
};

// Out-of-line definitions that need L0V2DeviceImageTy / L0V2DeviceTy.

Error L0V2GlobalHandlerTy::getGlobalMetadataFromDevice(
    GenericDeviceTy &Device, DeviceImageTy &Image, GlobalTy &DeviceGlobal) {
  auto *L0Image = static_cast<L0V2DeviceImageTy *>(&Image);
  ze_module_handle_t ZeModule = L0Image->getModule();

  size_t Size = 0;
  void *DevicePtr = nullptr;
  ze_result_t Res = zeModuleGetGlobalPointer(
      ZeModule, DeviceGlobal.getName().c_str(), &Size, &DevicePtr);
  if (Res != ZE_RESULT_SUCCESS)
    return Plugin::error(ErrorCode::NOT_FOUND,
                         "global '%s' not found in module",
                         DeviceGlobal.getName().c_str());

  if (DeviceGlobal.getSize() == 0)
    DeviceGlobal.setSize(Size);
  DeviceGlobal.setPtr(DevicePtr);
  return Plugin::success();
}

Error L0V2KernelTy::initImpl(GenericDeviceTy &GenericDevice,
                              DeviceImageTy &Image) {
  auto *L0Image = static_cast<L0V2DeviceImageTy *>(&Image);
  auto &Device = static_cast<L0V2DeviceTy &>(GenericDevice);
  ze_module_handle_t ZeModule = L0Image->getModule();

  // UR v2 adapter pattern: zeKernelCreate with kernel name.
  ze_kernel_desc_t KernelDesc{};
  KernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  KernelDesc.pKernelName = getName();

  ze_result_t ZeResult = zeKernelCreate(ZeModule, &KernelDesc, &ZeKernel);
  if (ZeResult == ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
    return Plugin::error(ErrorCode::NOT_FOUND, "kernel '%s' not found",
                         getName());
  if (ZeResult != ZE_RESULT_SUCCESS)
    return Plugin::error(ErrorCode::UNKNOWN,
                         "zeKernelCreate failed with error %d",
                         static_cast<int>(ZeResult));

  // Query kernel properties to get argument count.
  ze_kernel_properties_t KProps{};
  KProps.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  CALL_ZE_RET_ERROR(zeKernelGetProperties, ZeKernel, &KProps);
  NumKernelArgs = KProps.numKernelArgs;

  // Query per-argument sizes via zexKernelGetArgumentSize extension
  // (same approach as the L0 plugin).
  if (NumKernelArgs > 0) {
    using ZexKernelGetArgSizeFn = ze_result_t(ZE_APICALL *)(
        ze_kernel_handle_t, uint32_t, uint32_t *);
    ZexKernelGetArgSizeFn zexKernelGetArgumentSize = nullptr;
    ze_result_t ExtResult = zeDriverGetExtensionFunctionAddress(
        Device.ZeDriver, "zexKernelGetArgumentSize",
        reinterpret_cast<void **>(&zexKernelGetArgumentSize));
    if (ExtResult == ZE_RESULT_SUCCESS && zexKernelGetArgumentSize) {
      ArgSizes = std::make_unique<uint32_t[]>(NumKernelArgs);
      for (uint32_t I = 0; I < NumKernelArgs; I++) {
        CALL_ZE_RET_ERROR(zexKernelGetArgumentSize, ZeKernel, I,
                          &ArgSizes[I]);
      }
    }
  }

  return Plugin::success();
}

// Out-of-line definitions for L0V2KernelTy methods that need L0V2DeviceTy.

Error L0V2KernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                                uint32_t NumThreads[3], uint32_t NumBlocks[3],
                                uint32_t DynBlockMemSize,
                                KernelArgsTy &KernelArgs,
                                KernelLaunchParamsTy LaunchParams,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  auto &Device = static_cast<L0V2DeviceTy &>(GenericDevice);

  // Set kernel arguments from raw data.
  // Use user-provided ArgSizes if available, otherwise fall back to
  // cached sizes from zexKernelGetArgumentSize (L0 plugin pattern).
  if (LaunchParams.Data && LaunchParams.Size > 0 && NumKernelArgs > 0) {
    const char *ArgData = static_cast<const char *>(LaunchParams.Data);
    for (uint32_t I = 0; I < NumKernelArgs; I++) {
      uint32_t ArgSize;
      if (KernelArgs.ArgSizes)
        ArgSize = static_cast<uint32_t>(KernelArgs.ArgSizes[I]);
      else if (ArgSizes)
        ArgSize = ArgSizes[I];
      else
        return Plugin::error(
            ErrorCode::INVALID_ARGUMENT,
            "kernel argument sizes not provided and not available from driver");
      CALL_ZE_RET_ERROR(zeKernelSetArgumentValue, ZeKernel, I, ArgSize,
                        ArgData);
      ArgData += ArgSize;
    }
  }

  // UR v2 adapter pattern: set group size then launch.
  uint32_t GroupSizeX = NumThreads[0] ? NumThreads[0] : 1;
  uint32_t GroupSizeY = NumThreads[1] ? NumThreads[1] : 1;
  uint32_t GroupSizeZ = NumThreads[2] ? NumThreads[2] : 1;
  CALL_ZE_RET_ERROR(zeKernelSetGroupSize, ZeKernel, GroupSizeX, GroupSizeY,
                    GroupSizeZ);

  ze_group_count_t ZeGroupCount{};
  ZeGroupCount.groupCountX = NumBlocks[0] ? NumBlocks[0] : 1;
  ZeGroupCount.groupCountY = NumBlocks[1] ? NumBlocks[1] : 1;
  ZeGroupCount.groupCountZ = NumBlocks[2] ? NumBlocks[2] : 1;

  // Get the immediate command list from the async info.
  auto ImmCmdList = AsyncInfoWrapper.getQueueAs<ze_command_list_handle_t>();
  if (!ImmCmdList) {
    ze_command_queue_desc_t QueueDesc{};
    QueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    QueueDesc.ordinal = Device.ComputeQueueGroupOrdinal;
    QueueDesc.index = 0;
    QueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, Device.ZeContext,
                      Device.ZeDevice, &QueueDesc, &ImmCmdList);
    AsyncInfoWrapper.setQueueAs<ze_command_list_handle_t>(ImmCmdList);
  }

  // UR v2 adapter: cooperative vs regular launch.
  if (KernelArgs.Flags.Cooperative) {
    // Validate group count against device maximum (the driver may not reject
    // excessive counts on all hardware).
    uint32_t MaxCoopGroups = 0;
    CALL_ZE_RET_ERROR(zeKernelSuggestMaxCooperativeGroupCount, ZeKernel,
                      &MaxCoopGroups);
    uint32_t TotalGroups = ZeGroupCount.groupCountX *
                           ZeGroupCount.groupCountY *
                           ZeGroupCount.groupCountZ;
    if (TotalGroups > MaxCoopGroups)
      return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                           "cooperative launch requested %u groups but device "
                           "supports at most %u",
                           TotalGroups, MaxCoopGroups);

    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchCooperativeKernel, ImmCmdList,
                      ZeKernel, &ZeGroupCount, nullptr, 0, nullptr);
  } else {
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, ImmCmdList, ZeKernel,
                      &ZeGroupCount, nullptr, 0, nullptr);
  }

  return Plugin::success();
}

Expected<uint64_t>
L0V2KernelTy::maxGroupSize(GenericDeviceTy &GenericDevice,
                            uint64_t DynamicMemSize) const {
  auto &Device = static_cast<L0V2DeviceTy &>(GenericDevice);
  return static_cast<uint64_t>(
      Device.ZeDeviceComputeProperties.maxTotalGroupSize);
}

Expected<uint32_t> L0V2KernelTy::getMaxCooperativeGroupCount(
    GenericDeviceTy &GenericDevice, const uint32_t NumThreads[3],
    uint32_t DynBlockMemSize) const {
  // UR v2 adapter pattern: set group size then query max cooperative count.
  CALL_ZE_RET_ERROR(zeKernelSetGroupSize, ZeKernel,
                    NumThreads[0] ? NumThreads[0] : 1,
                    NumThreads[1] ? NumThreads[1] : 1,
                    NumThreads[2] ? NumThreads[2] : 1);

  uint32_t TotalGroupCount = 0;
  CALL_ZE_RET_ERROR(zeKernelSuggestMaxCooperativeGroupCount, ZeKernel,
                    &TotalGroupCount);
  return TotalGroupCount;
}

/// Plugin implementation for the Level Zero V2 plugin.
struct L0V2PluginTy final : public GenericPluginTy {
  /// Per-driver context information.
  struct DriverContextTy {
    ze_driver_handle_t ZeDriver;
    ze_context_handle_t ZeContext;
  };

  /// Per-device information collected during plugin init.
  struct DeviceInfoTy {
    ze_device_handle_t ZeDevice;
    ze_context_handle_t ZeContext;
    ze_driver_handle_t ZeDriver;
  };

  /// Contexts created for each driver.
  llvm::SmallVector<DriverContextTy> DriverContexts;

  /// Detected devices.
  llvm::SmallVector<DeviceInfoTy> DetectedDevices;

  L0V2PluginTy() : GenericPluginTy(getTripleArch()) {}

  Expected<int32_t> initImpl() override {
    ODBG(OLDT_Init) << "Level Zero V2 plugin initialization";

    // Following UR v2 adapter pattern: zeInit with GPU_ONLY flag.
    ze_init_flags_t L0InitFlags = ZE_INIT_FLAG_GPU_ONLY;
    CALL_ZE_RET_ERROR(zeInit, L0InitFlags);

    // Query available drivers (UR v2: zeDriverGet two-step pattern).
    uint32_t ZeDriverCount = 0;
    ze_result_t ZeResult = zeDriverGet(&ZeDriverCount, nullptr);
    if (ZeResult != ZE_RESULT_SUCCESS || ZeDriverCount == 0) {
      ODBG(OLDT_Init) << "No Level Zero drivers found.";
      return 0;
    }

    llvm::SmallVector<ze_driver_handle_t> ZeDrivers(ZeDriverCount);
    CALL_ZE_RET_ERROR(zeDriverGet, &ZeDriverCount, ZeDrivers.data());
    ODBG(OLDT_Init) << ZeDriverCount << " L0 driver(s) found.";

    // Following UR v2 adapter pattern (initPlatforms):
    // For each driver, enumerate devices and filter to GPU type only.
    for (uint32_t I = 0; I < ZeDriverCount; I++) {
      ze_driver_handle_t ZeDriver = ZeDrivers[I];
      uint32_t ZeDeviceCount = 0;
      CALL_ZE_RET_ERROR(zeDeviceGet, ZeDriver, &ZeDeviceCount, nullptr);
      if (ZeDeviceCount == 0)
        continue;

      llvm::SmallVector<ze_device_handle_t> ZeDevices(ZeDeviceCount);
      CALL_ZE_RET_ERROR(zeDeviceGet, ZeDriver, &ZeDeviceCount,
                        ZeDevices.data());

      // Check if this driver has any GPU devices (UR v2 adapter pattern).
      bool HasGPU = false;
      for (uint32_t D = 0; D < ZeDeviceCount; D++) {
        if (!isDeviceGPU(ZeDevices[D]))
          continue;

        // Create context for this driver on first GPU device found
        // (UR v2 adapter: context created per-platform/driver).
        if (!HasGPU) {
          ze_context_desc_t CtxDesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
          ze_context_handle_t ZeContext = nullptr;
          CALL_ZE_RET_ERROR(zeContextCreate, ZeDriver, &CtxDesc, &ZeContext);
          DriverContexts.push_back({ZeDriver, ZeContext});
          HasGPU = true;
        }

        DetectedDevices.push_back(
            {ZeDevices[D], DriverContexts.back().ZeContext, ZeDriver});
      }
    }

    int32_t NumDevices = DetectedDevices.size();
    ODBG(OLDT_Init) << "Found " << NumDevices << " Level Zero GPU device(s).";

    return NumDevices;
  }

  Error deinitImpl() override {
    // Destroy all Level Zero contexts.
    for (auto &DC : DriverContexts) {
      if (DC.ZeContext) {
        ze_result_t RC = zeContextDestroy(DC.ZeContext);
        if (RC != ZE_RESULT_SUCCESS)
          ODBG(OLDT_Deinit) << "zeContextDestroy failed with error "
                            << static_cast<int>(RC);
      }
    }
    DriverContexts.clear();
    DetectedDevices.clear();
    return Plugin::success();
  }

  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    auto &Info = DetectedDevices[DeviceId];
    return new L0V2DeviceTy(Plugin, DeviceId, NumDevices, Info.ZeDevice,
                            Info.ZeContext, Info.ZeDriver);
  }

  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new L0V2GlobalHandlerTy();
  }

  uint16_t getMagicElfBits() const override { return ELF::EM_INTELGT; }
  Triple::ArchType getTripleArch() const override { return Triple::spirv64; }
  const char *getName() const override { return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    // Accept IntelGT ELF binaries (native binaries from the driver).
    return true;
  }

  /// Accept SPIR-V and OffloadBinary formats directly.
  Expected<bool> isImageCompatible(StringRef Image) const override {
    switch (identify_magic(Image)) {
    case file_magic::spirv_object:
      return true;
    case file_magic::offload_binary: {
      auto BinsOrErr = object::OffloadBinary::create(
          MemoryBufferRef(Image, "offload_binary"));
      if (!BinsOrErr)
        return BinsOrErr.takeError();
      if (BinsOrErr->size() != 1)
        return false;
      auto *Inner = (*BinsOrErr)[0].get();
      return llvm::Triple(Inner->getTriple()).getArch() == Triple::spirv64;
    }
    default:
      return false;
    }
  }
};

} // namespace llvm::omp::target::plugin

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_level_zero_v2() {
  return new llvm::omp::target::plugin::L0V2PluginTy();
}
}
