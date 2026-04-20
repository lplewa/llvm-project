//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GenericDevice instatiation for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#include "L0Device.h"
#include "L0Defs.h"
#include "L0Interop.h"
#include "L0Plugin.h"
#include "L0Program.h"
#include "L0Trace.h"

#include "GlobalHandler.h"
#include "OffloadAPI.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/ELF.h"

namespace llvm::omp::target::plugin {

// clang-format off
/// Mapping from device arch to GPU runtime's device identifiers.
static struct {
  DeviceArchTy arch;
  PCIIdTy ids[10];
} DeviceArchMap[] = {{DeviceArchTy::DeviceArch_Gen,
                      {PCIIdTy::SKL,
                       PCIIdTy::KBL,
                       PCIIdTy::CFL, PCIIdTy::CFL_2,
                       PCIIdTy::ICX,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Gen,
                      {PCIIdTy::TGL, PCIIdTy::TGL_2,
                       PCIIdTy::DG1,
                       PCIIdTy::RKL,
                       PCIIdTy::ADLS,
                       PCIIdTy::RTL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeLPG,
                      {PCIIdTy::MTL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeHPC,
                      {PCIIdTy::PVC,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeHPG,
                      {PCIIdTy::DG2_ATS_M,
                       PCIIdTy::DG2_ATS_M_2,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Xe2LP,
                      {PCIIdTy::LNL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Xe2HP,
                      {PCIIdTy::BMG,
                       PCIIdTy::None}},
};
constexpr int DeviceArchMapSize = sizeof(DeviceArchMap) / sizeof(DeviceArchMap[0]);
// clang-format on

DeviceArchTy L0DeviceTy::computeArch() const {
  const auto PCIDeviceId = getPCIId();
  if (PCIDeviceId == 0) {
    ODBG(OLDT_Device) << "Warning: Cannot decide device arch for " << getName()
                      << ".";
    return DeviceArchTy::DeviceArch_None;
  }

  for (int ArchIndex = 0; ArchIndex < DeviceArchMapSize; ArchIndex++) {
    for (int i = 0;; i++) {
      const auto Id = DeviceArchMap[ArchIndex].ids[i];
      if (Id == PCIIdTy::None)
        break;
      auto maskedId = static_cast<PCIIdTy>(PCIDeviceId & 0xFF00);
      if (maskedId == Id)
        return DeviceArchMap[ArchIndex].arch; // Exact match or prefix match.
    }
  }

  ODBG(OLDT_Device) << "Warning: Cannot decide device arch for " << getName()
                    << ".";
  return DeviceArchTy::DeviceArch_None;
}

/// Get default compute group ordinal. Returns Ordinal-NumQueues pair.
std::pair<uint32_t, uint32_t> L0DeviceTy::findComputeOrdinal() {
  std::pair<uint32_t, uint32_t> Ordinal{MaxOrdinal, 0};
  uint32_t Count = 0;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              nullptr);
  ze_command_queue_group_properties_t Init{
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
  std::vector<ze_command_queue_group_properties_t> Properties(Count, Init);
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              Properties.data());
  for (uint32_t I = 0; I < Count; I++) {
    // TODO: add a separate set of ordinals for compute queue groups which
    // support cooperative kernels.
    if (Properties[I].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      Ordinal.first = I;
      Ordinal.second = Properties[I].numQueues;
      break;
    }
  }
  if (Ordinal.first == MaxOrdinal)
    ODBG(OLDT_Device) << "Error: no command queues are found";

  return Ordinal;
}

void L0DeviceTy::reportDeviceInfo() const {
  ODBG_OS(OLDT_Device, [&](llvm::raw_ostream &O) {
    O << "Device " << DeviceId << " information\n"
      << "-- Name                         : " << getName() << "\n"
      << "-- PCI ID                       : "
      << llvm::format("0x%" PRIx32, getPCIId()) << "\n"
      << "-- UUID                         : " << getUuid().data() << "\n"
      << "-- Number of total EUs          : " << getNumEUs() << "\n"
      << "-- Number of threads per EU     : " << getNumThreadsPerEU() << "\n"
      << "-- EU SIMD width                : " << getSIMDWidth() << "\n"
      << "-- Number of EUs per subslice   : " << getNumEUsPerSubslice() << "\n"
      << "-- Number of subslices per slice: " << getNumSubslicesPerSlice()
      << "\n"
      << "-- Number of slices             : " << getNumSlices() << "\n"
      << "-- Local memory size (bytes)    : " << getMaxSharedLocalMemory()
      << "\n"
      << "-- Global memory size (bytes)   : " << getGlobalMemorySize() << "\n"
      << "-- Cache size (bytes)           : " << getCacheSize() << "\n"
      << "-- Max clock frequency (MHz)    : " << getClockRate() << "\n";
  });
}

Error L0DeviceTy::initImpl(GenericPluginTy &Plugin) {
  const auto &Options = getPlugin().getOptions();

  uint32_t Count = 1;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET_ERROR(zeDeviceGetProperties, zeDevice, &DeviceProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetComputeProperties, zeDevice, &ComputeProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetMemoryProperties, zeDevice, &Count,
                    &MemoryProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetCacheProperties, zeDevice, &Count,
                    &CacheProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetModuleProperties, zeDevice, &ModuleProperties);

  DeviceName = std::string(DeviceProperties.name);

  ODBG(OLDT_Device) << "Found a GPU device, Name = " << DeviceProperties.name;

  DeviceArch = computeArch();
  // Default allocation kind for this device.
  AllocKind = isDiscreteDevice() ? TARGET_ALLOC_DEVICE : TARGET_ALLOC_SHARED;

  ze_kernel_indirect_access_flags_t Flags =
      (AllocKind == TARGET_ALLOC_DEVICE)
          ? ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE
          : ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
  IndirectAccessFlags = Flags;

  // Get the UUID.
  std::string uid;
  for (int n = 0; n < ZE_MAX_DEVICE_UUID_SIZE; n++)
    uid += std::to_string(DeviceProperties.uuid.id[n]);
  DeviceUuid = std::move(uid);

  ComputeOrdinal = findComputeOrdinal();

  if (auto Err = MemAllocator.initDevicePools(*this, Options))
    return Err;
  l0Context.getHostMemAllocator().updateMaxAllocSize(*this);
  reportDeviceInfo();
  return Plugin::success();
}

Error L0DeviceTy::deinitImpl() {
  Error AllErrors = Error::success();
  for (auto &PGM : Programs)
    if (auto Err = PGM.deinit())
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
  for (auto CmdList : ImmCmdListCache)
    CALL_ZE_ACCUM_ERROR(AllErrors, zeCommandListDestroy, CmdList);
  ImmCmdListCache.clear();
  if (auto Err = MemAllocator.deinit())
    AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
  return AllErrors;
}

Expected<DeviceImageTy *>
L0DeviceTy::loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                           int32_t ImageId) {
  auto *PGM = getProgramFromImage(TgtImage->getMemBufferRef());
  if (PGM) {
    // Program already exists.
    return PGM;
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Device %" PRId32 ": Loading binary from " DPxMOD "\n", getDeviceId(),
       DPxPTR(TgtImage->getBufferStart()));

  const auto &Options = getPlugin().getOptions();
  std::string CompilationOptions(Options.CompilationOptions);
  CompilationOptions += " " + Options.UserCompilationOptions;

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Base L0 module compilation options: %s\n", CompilationOptions.c_str());

  CompilationOptions += " ";
  CompilationOptions += Options.InternalCompilationOptions;

  L0ProgramBuilderTy Builder(*this, std::move(TgtImage));
  if (auto Err = Builder.buildModules(CompilationOptions))
    return std::move(Err);

  auto ProgramOrErr = addProgram(ImageId, Builder);
  if (!ProgramOrErr)
    return ProgramOrErr.takeError();
  auto &Program = *ProgramOrErr;

  if (auto Err = Program.loadModuleKernels())
    return std::move(Err);

  return &Program;
}

Error L0DeviceTy::unloadBinaryImpl(DeviceImageTy *Image) {
  // Ignoring for now.
  // TODO: call properly L0Program unload.
  return Plugin::success();
}

namespace {
class L0ImmInOrderQueueTy final : public L0AsyncQueueTy {
  L0DeviceTy &Device;
  ze_command_list_handle_t ImmCmdList;

public:
  L0ImmInOrderQueueTy(L0DeviceTy &Device, ze_command_list_handle_t CmdList)
      : Device(Device), ImmCmdList(CmdList) {}

  Error appendKernelLaunch(ze_kernel_handle_t Kernel,
                           const ze_group_count_t &Groups,
                           ze_event_handle_t SignalEvent) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, ImmCmdList, Kernel,
                      &Groups, SignalEvent, 0, nullptr);
    return Plugin::success();
  }

  Error appendKernelLaunchWithArguments(
      ze_kernel_handle_t Kernel, const ze_group_count_t &Groups,
      const ze_group_size_t &GroupSizes, void **ArgPtrs,
      ze_event_handle_t SignalEvent) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernelWithArguments, ImmCmdList,
                      Kernel, Groups, GroupSizes, ArgPtrs, nullptr, SignalEvent,
                      0, nullptr);
    return Plugin::success();
  }

  Error appendMemCopy(void *Dst, const void *Src, size_t Size,
                      ze_event_handle_t SignalEvent) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, ImmCmdList, Dst, Src,
                      Size, SignalEvent, 0, nullptr);
    return Plugin::success();
  }

  Error appendMemFill(void *Ptr, const void *Pattern, size_t PatternSize,
                      size_t Size, ze_event_handle_t SignalEvent) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, ImmCmdList, Ptr, Pattern,
                      PatternSize, Size, SignalEvent, 0, nullptr);
    return Plugin::success();
  }

  Error appendSignalEvent(ze_event_handle_t Event) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendSignalEvent, ImmCmdList, Event);
    return Plugin::success();
  }

  Error appendWaitOnEvent(ze_event_handle_t Event) override {
    CALL_ZE_RET_ERROR(zeCommandListAppendWaitOnEvents, ImmCmdList, 1, &Event);
    return Plugin::success();
  }

  Error hostSynchronize(uint64_t TimeoutNs) override {
    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList, TimeoutNs);
    return Plugin::success();
  }

  Expected<bool> isComplete() override {
    ze_result_t RC = zeCommandListHostSynchronize(ImmCmdList, 0);
    if (RC == ZE_RESULT_SUCCESS)
      return true;
    if (RC == ZE_RESULT_NOT_READY)
      return false;
    return Plugin::error(ErrorCode::UNKNOWN,
                         "zeCommandListHostSynchronize query failed: 0x%x",
                         RC);
  }

  Error destroy() override {
    Device.releaseImmCmdList(ImmCmdList);
    ImmCmdList = nullptr;
    return Plugin::success();
  }
};

static Error destroyQueue(__tgt_async_info &AsyncInfo) {
  auto *Queue = static_cast<L0AsyncQueueTy *>(AsyncInfo.Queue);
  if (!Queue)
    return Plugin::success();
  Error Err = Queue->destroy();
  delete Queue;
  AsyncInfo.Queue = nullptr;
  return Err;
}
} // namespace

Error L0DeviceTy::synchronizeImpl(__tgt_async_info &AsyncInfo,
                                  bool ReleaseQueue) {
  if (!AsyncInfo.Queue)
    return Plugin::success();

  auto *Queue = static_cast<L0AsyncQueueTy *>(AsyncInfo.Queue);

  Error SyncErr = Queue->hostSynchronize(L0DefaultTimeout);

  if (ReleaseQueue) {
    if (auto Err = destroyQueue(AsyncInfo))
      SyncErr = joinErrors(std::move(SyncErr), std::move(Err));
  }
  return SyncErr;
}

Expected<bool>
L0DeviceTy::hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  auto &AsyncInfo = *static_cast<__tgt_async_info *>(AsyncInfoWrapper);
  if (!AsyncInfo.Queue)
    return false;
  auto *Queue = static_cast<L0AsyncQueueTy *>(AsyncInfo.Queue);
  auto CompleteOrErr = Queue->isComplete();
  if (!CompleteOrErr)
    return CompleteOrErr.takeError();
  return !*CompleteOrErr;
}

Error L0DeviceTy::queryAsyncImpl(__tgt_async_info &AsyncInfo, bool ReleaseQueue,
                                 bool *IsQueueWorkCompleted) {
  if (IsQueueWorkCompleted)
    *IsQueueWorkCompleted = true;
  if (!AsyncInfo.Queue)
    return Plugin::success();

  auto *Queue = static_cast<L0AsyncQueueTy *>(AsyncInfo.Queue);
  auto CompleteOrErr = Queue->isComplete();
  if (!CompleteOrErr)
    return CompleteOrErr.takeError();

  if (IsQueueWorkCompleted)
    *IsQueueWorkCompleted = *CompleteOrErr;
  if (!*CompleteOrErr)
    return Plugin::success();

  if (ReleaseQueue)
    return destroyQueue(AsyncInfo);
  return Plugin::success();
}

Expected<void *> L0DeviceTy::allocate(size_t Size, void *HstPtr,
                                      TargetAllocTy Kind) {
  return dataAlloc(Size, /*Align=*/0, Kind,
                   /*Offset=*/0, /*UserAlloc=*/HstPtr == nullptr,
                   /*DevMalloc=*/false);
}

Error L0DeviceTy::free(void *TgtPtr, TargetAllocTy Kind) {
  return dataDelete(TgtPtr);
}

Expected<L0AsyncQueueTy *>
L0DeviceTy::getOrCreateQueue(__tgt_async_info *AsyncInfo) {
  if (auto *Existing = static_cast<L0AsyncQueueTy *>(AsyncInfo->Queue))
    return Existing;
  auto CmdListOrErr = acquireImmCmdList();
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  auto *Queue = new L0ImmInOrderQueueTy(*this, *CmdListOrErr);
  AsyncInfo->Queue = Queue;
  return static_cast<L0AsyncQueueTy *>(Queue);
}

Error L0DeviceTy::dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (Size == 0)
    return Plugin::success();

  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  if (auto Err = (*QueueOrErr)->appendMemCopy(TgtPtr, HstPtr, Size, nullptr))
    return Err;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Submitted copy %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
       Size, DPxPTR(HstPtr), DPxPTR(TgtPtr));
  return Plugin::success();
}

Error L0DeviceTy::dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                   int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (Size == 0)
    return Plugin::success();

  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  if (auto Err = (*QueueOrErr)->appendMemCopy(HstPtr, TgtPtr, Size, nullptr))
    return Err;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Submitted copy %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
       Size, DPxPTR(TgtPtr), DPxPTR(HstPtr));
  return Plugin::success();
}

Error L0DeviceTy::dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                   void *DstPtr, int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  return (*QueueOrErr)->appendMemCopy(DstPtr, SrcPtr, Size, nullptr);
}

Error L0DeviceTy::initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  return getOrCreateQueue(AsyncInfo).takeError();
}

const char *L0DeviceTy::getArchCStr() const {
  switch (getDeviceArch()) {
  case DeviceArchTy::DeviceArch_Gen:
    return "Intel GPU Xe";
  case DeviceArchTy::DeviceArch_XeLPG:
    return "Intel GPU Xe LPG";
  case DeviceArchTy::DeviceArch_XeHPC:
    return "Intel GPU Xe HPC";
  case DeviceArchTy::DeviceArch_XeHPG:
    return "Intel GPU Xe HPG";
  case DeviceArchTy::DeviceArch_Xe2LP:
    return "Intel GPU Xe2 LP";
  case DeviceArchTy::DeviceArch_Xe2HP:
    return "Intel GPU Xe HP";
  case DeviceArchTy::DeviceArch_x86_64:
    return "Intel X86 64";
  default:
    return "Intel GPU Unknown";
  }
}

static const char *DriverVersionToStrTable[] = {
    "1.0", "1.1", "1.2", "1.3",  "1.4",  "1.5", "1.6",
    "1.7", "1.8", "1.9", "1.10", "1.11", "1.12"};
constexpr size_t DriverVersionToStrTableSize =
    sizeof(DriverVersionToStrTable) / sizeof(DriverVersionToStrTable[0]);

Expected<InfoTreeNode> L0DeviceTy::obtainInfoImpl() {
  InfoTreeNode Info;
  Info.add("Device Number", getDeviceId());
  Info.add("Device Name", getNameCStr(), "", DeviceInfo::NAME);
  Info.add("Product Name", getArchCStr(), "", DeviceInfo::PRODUCT_NAME);
  Info.add("Device Type", "GPU", "", DeviceInfo::TYPE);
  Info.add("Vendor", "Intel", "", DeviceInfo::VENDOR);
  Info.add("Vendor ID", getVendorId(), "", DeviceInfo::VENDOR_ID);
  auto DriverVersion = getDriverAPIVersion();
  if (DriverVersion < DriverVersionToStrTableSize)
    Info.add("Driver Version", DriverVersionToStrTable[DriverVersion], "",
             DeviceInfo::DRIVER_VERSION);
  else
    Info.add("Driver Version", "Unknown", "", DeviceInfo::DRIVER_VERSION);
  Info.add("Device PCI ID", getPCIId());
  Info.add("Device UUID", getUuid().data());
  Info.add("Number of total EUs", getNumEUs(), "",
           DeviceInfo::NUM_COMPUTE_UNITS);
  Info.add("Number of threads per EU", getNumThreadsPerEU());
  Info.add("EU SIMD width", getSIMDWidth());
  Info.add("Number of EUs per subslice", getNumEUsPerSubslice());
  Info.add("Number of subslices per slice", getNumSubslicesPerSlice());
  Info.add("Number of slices", getNumSlices());
  Info.add("Max Group size", getMaxGroupSize(), "",
           DeviceInfo::MAX_WORK_GROUP_SIZE);
  auto &MaxGroupSize =
      *Info.add("Workgroup Max Size per Dimension", std::monostate{}, "",
                DeviceInfo::MAX_WORK_GROUP_SIZE_PER_DIMENSION);
  MaxGroupSize.add("x", getMaxGroupSizeX());
  MaxGroupSize.add("y", getMaxGroupSizeY());
  MaxGroupSize.add("z", getMaxGroupSizeZ());
  Info.add("Maximum Grid Dimensions", getMaxGroupSize() * getMaxGroupCount(),
           "", DeviceInfo::MAX_WORK_SIZE);
  auto &MaxSize = *Info.add("Grid Size per Dimension", std::monostate{}, "",
                            DeviceInfo::MAX_WORK_SIZE_PER_DIMENSION);
  MaxSize.add("x", getMaxGroupSizeX() * getMaxGroupCountX());
  MaxSize.add("y", getMaxGroupSizeY() * getMaxGroupCountY());
  MaxSize.add("z", getMaxGroupSizeZ() * getMaxGroupCountZ());

  Info.add("Local memory size (bytes)", getMaxSharedLocalMemory(), "",
           DeviceInfo::WORK_GROUP_LOCAL_MEM_SIZE);
  Info.add("Global memory size (bytes)", getGlobalMemorySize(), "",
           DeviceInfo::GLOBAL_MEM_SIZE);
  Info.add("Cache size (bytes)", getCacheSize());
  Info.add("Max Memory Allocation Size (bytes)", getMaxMemAllocSize(), "",
           DeviceInfo::MAX_MEM_ALLOC_SIZE);
  Info.add("Max clock frequency (MHz)", getClockRate(), "",
           DeviceInfo::MAX_CLOCK_FREQUENCY);
  Info.add("Max memory clock frequency (MHz)", getMemoryClockRate(), "",
           DeviceInfo::MEMORY_CLOCK_RATE);
  Info.add("Memory Address Size", uint64_t{64u}, "bits",
           DeviceInfo::ADDRESS_BITS);

  // FP64 (Double precision).
  Info.add("Double FP Support", supportsFP64(), "",
           DeviceInfo::DOUBLE_FP_SUPPORT);
  ol_device_fp_capability_flags_t DoubleFPCapabilities = 0;
  ze_device_fp_flags_t ZeDoubleFPFlags = getFP64Flags();
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    DoubleFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Double FP Capabilities", DoubleFPCapabilities, "",
           DeviceInfo::DOUBLE_FP_CONFIG);

  // FP16 (Half precision).
  Info.add("Half FP Support", supportsFP16(), "", DeviceInfo::HALF_FP_SUPPORT);
  ol_device_fp_capability_flags_t HalfFPCapabilities = 0;
  ze_device_fp_flags_t ZeHalfFPFlags = getFP16Flags();
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    HalfFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Half FP Capabilities", HalfFPCapabilities, "",
           DeviceInfo::HALF_FP_CONFIG);

  // FP32 (Single FP).
  Info.add("Single FP Support", true, "", DeviceInfo::SINGLE_FP_SUPPORT);
  ol_device_fp_capability_flags_t SingleFPCapabilities = 0;
  ze_device_fp_flags_t ZeSingleFPFlags = getFP32Flags();
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    SingleFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Single FP Capabilities", SingleFPCapabilities, "",
           DeviceInfo::SINGLE_FP_CONFIG);

  return Info;
}

Expected<GenericKernelTy &> L0DeviceTy::constructKernel(const char *Name) {
  // Allocate and construct the L0 kernel.
  L0KernelTy *L0Kernel = getPlugin().allocate<L0KernelTy>();
  if (!L0Kernel)
    return Plugin::error(ErrorCode::UNKNOWN,
                         "Failed to allocate memory for L0 kernel");

  new (L0Kernel) L0KernelTy(Name);

  return *L0Kernel;
}

uint32_t L0DeviceTy::getMemAllocType(const void *Ptr) const {
  ze_memory_allocation_properties_t properties = {
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
      nullptr,                // Extension.
      ZE_MEMORY_TYPE_UNKNOWN, // Type.
      0,                      // Id.
      0,                      // Page size.
  };

  ze_result_t rc;
  CALL_ZE(rc, zeMemGetAllocProperties, getZeContext(), Ptr, &properties,
          nullptr);

  if (rc == ZE_RESULT_ERROR_INVALID_ARGUMENT)
    return ZE_MEMORY_TYPE_UNKNOWN;
  else
    return properties.type;
}

interop_spec_t L0DeviceTy::selectInteropPreference(int32_t InteropType,
                                                   int32_t NumPrefers,
                                                   interop_spec_t *Prefers) {
  // no supported preference found, set default to level_zero,
  // non-ordered unless is targetsync.
  return interop_spec_t{
      tgt_fr_level_zero,
      {InteropType == kmp_interop_type_targetsync /*inorder*/, 0},
      0};
}

Expected<OmpInteropTy> L0DeviceTy::createInterop(int32_t InteropContext,
                                                 interop_spec_t &InteropSpec) {
  auto Ret = new omp_interop_val_t(
      DeviceId, static_cast<kmp_interop_type_t>(InteropContext));
  Ret->fr_id = tgt_fr_level_zero;
  Ret->vendor_id = omp_vendor_intel;

  if (InteropContext == kmp_interop_type_target ||
      InteropContext == kmp_interop_type_targetsync) {
    Ret->device_info.Platform = getZeDriver();
    Ret->device_info.Device = getZeDevice();
    Ret->device_info.Context = getZeContext();
  }

  Ret->rtl_property = new L0Interop::Property();
  if (InteropContext == kmp_interop_type_targetsync) {
    Ret->async_info = new __tgt_async_info();

    llvm::scope_exit CleanupOnError([&]() {
      if (Ret->async_info)
        delete Ret->async_info;
      if (Ret->rtl_property)
        delete static_cast<L0Interop::Property *>(Ret->rtl_property);
      delete Ret;
    });

    auto L0 = static_cast<L0Interop::Property *>(Ret->rtl_property);

    // Only immediate in-order queue is supported.
    Ret->attrs.inorder = 1;
    auto CmdListOrErr = createImmCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    Ret->async_info->Queue = *CmdListOrErr;
    L0->ImmCmdList = *CmdListOrErr;

    CleanupOnError.release();
  }

  return Ret;
}

Error L0DeviceTy::releaseInterop(OmpInteropTy Interop) {
  const auto DeviceId = getDeviceId();

  if (!Interop || Interop->device_id != static_cast<intptr_t>(DeviceId)) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid/inconsistent OpenMP interop " DPxMOD "\n",
                         DPxPTR(Interop));
  }
  auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);
  if (Interop->async_info && Interop->async_info->Queue) {
    // Not cached: the user may have appended arbitrary commands.
    CALL_ZE_RET_ERROR(zeCommandListDestroy, L0->ImmCmdList);
  }
  delete Interop->async_info;
  delete L0;
  delete Interop;

  return Plugin::success();
}

Error L0DeviceTy::enqueueMemCopy(void *Dst, const void *Src, size_t Size) {
  auto CmdListOrErr = acquireImmCmdList();
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  ze_command_list_handle_t CmdList = *CmdListOrErr;
  llvm::scope_exit ReleaseOnExit([&]() { releaseImmCmdList(CmdList); });
  CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                    nullptr, 0, nullptr);
  CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, CmdList, L0DefaultTimeout);
  return Plugin::success();
}

Error L0DeviceTy::enqueueMemFill(void *Ptr, const void *Pattern,
                                 size_t PatternSize, size_t Size) {
  auto CmdListOrErr = acquireImmCmdList();
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  ze_command_list_handle_t CmdList = *CmdListOrErr;
  llvm::scope_exit ReleaseOnExit([&]() { releaseImmCmdList(CmdList); });
  CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, CmdList, Ptr, Pattern,
                    PatternSize, Size, nullptr, 0, nullptr);
  CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, CmdList, L0DefaultTimeout);
  return Plugin::success();
}

Error L0DeviceTy::dataFillImpl(void *TgtPtr, const void *PatternPtr,
                               int64_t PatternSize, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  return (*QueueOrErr)->appendMemFill(TgtPtr, PatternPtr, PatternSize, Size,
                                       nullptr);
}

Expected<void *> L0DeviceTy::dataAlloc(size_t Size, size_t Align, int32_t Kind,
                                       intptr_t Offset, bool UserAlloc,
                                       bool DevMalloc, uint32_t MemAdvice,
                                       AllocOptionTy AllocOpt) {

  const bool UseDedicatedPool =
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_SCRATCH) ||
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_COUNTER);
  if (Kind == TARGET_ALLOC_DEFAULT) {
    if (UserAlloc)
      Kind = TARGET_ALLOC_DEVICE;
    else if (AllocOpt == AllocOptionTy::ALLOC_OPT_HOST_MEM)
      Kind = TARGET_ALLOC_HOST;
    else if (UseDedicatedPool)
      Kind = TARGET_ALLOC_DEVICE;
    else
      Kind = getAllocKind();
  }
  auto &Allocator = getMemAllocator(Kind);
  return Allocator.alloc(Size, Align, Kind, Offset, UserAlloc, DevMalloc,
                         MemAdvice, AllocOpt);
}

Error L0DeviceTy::dataDelete(void *Ptr) {
  auto &Allocator = getMemAllocator(Ptr);
  return Allocator.dealloc(Ptr);
}

Error L0DeviceTy::makeMemoryResident(void *Mem, size_t Size) {
  CALL_ZE_RET_ERROR(zeContextMakeMemoryResident, getZeContext(), getZeDevice(),
                    Mem, Size);
  return Plugin::success();
}

Expected<ze_command_list_handle_t> L0DeviceTy::createImmCmdList() {
  ze_command_queue_desc_t Desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                               nullptr,
                               getComputeEngine(),
                               getComputeIndex(),
                               ZE_COMMAND_QUEUE_FLAG_IN_ORDER |
                                   ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                               ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t CmdList = nullptr;
  CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, getZeContext(), getZeDevice(),
                    &Desc, &CmdList);
  ODBG(OLDT_Device) << "Created immediate command list " << CmdList
                    << " for device " << getZeIdCStr();
  return CmdList;
}

Expected<ze_command_list_handle_t> L0DeviceTy::acquireImmCmdList() {
  {
    std::lock_guard<std::mutex> Lock(ImmCmdListCacheMtx);
    if (!ImmCmdListCache.empty()) {
      ze_command_list_handle_t CmdList = ImmCmdListCache.back();
      ImmCmdListCache.pop_back();
      return CmdList;
    }
  }
  return createImmCmdList();
}

void L0DeviceTy::releaseImmCmdList(ze_command_list_handle_t CmdList) {
  if (!CmdList)
    return;
  std::lock_guard<std::mutex> Lock(ImmCmdListCacheMtx);
  ImmCmdListCache.push_back(CmdList);
}

Error L0DeviceTy::createEventImpl(void **EventPtrStorage) {
  auto EventOrErr = getEvent();
  if (!EventOrErr)
    return EventOrErr.takeError();
  *EventPtrStorage = *EventOrErr;
  return Plugin::success();
}

Error L0DeviceTy::destroyEventImpl(void *EventPtr) {
  return releaseEvent(static_cast<ze_event_handle_t>(EventPtr));
}

Error L0DeviceTy::recordEventImpl(void *EventPtr,
                                  AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  return (*QueueOrErr)
      ->appendSignalEvent(static_cast<ze_event_handle_t>(EventPtr));
}

Error L0DeviceTy::waitEventImpl(void *EventPtr,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  return (*QueueOrErr)
      ->appendWaitOnEvent(static_cast<ze_event_handle_t>(EventPtr));
}

Error L0DeviceTy::syncEventImpl(void *EventPtr) {
  CALL_ZE_RET_ERROR(zeEventHostSynchronize,
                    static_cast<ze_event_handle_t>(EventPtr), L0DefaultTimeout);
  return Plugin::success();
}

Expected<bool> L0DeviceTy::isEventCompleteImpl(void *EventPtr,
                                               AsyncInfoWrapperTy &) {
  ze_result_t RC =
      zeEventQueryStatus(static_cast<ze_event_handle_t>(EventPtr));
  if (RC == ZE_RESULT_SUCCESS)
    return true;
  if (RC == ZE_RESULT_NOT_READY)
    return false;
  return Plugin::error(ErrorCode::UNKNOWN,
                       "zeEventQueryStatus failed: 0x%x", RC);
}

Expected<float> L0DeviceTy::getEventElapsedTimeImpl(void *, void *) {
  return Plugin::error(ErrorCode::UNIMPLEMENTED,
                       "%s requires a timestamp-capable event pool",
                       __func__);
}

Expected<bool> L0DeviceTy::isAccessiblePtrImpl(const void *Ptr, size_t Size) {
  if (!Ptr || Size == 0)
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid input to %s (Ptr = %p, Size = %zu)", __func__,
                         Ptr, Size);
  return getMemAllocator(Ptr).contains(Ptr, Size);
}

Error L0DeviceTy::callGlobalConstructors(GenericPluginTy &Plugin,
                                         DeviceImageTy &Image) {
  return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
}

Error L0DeviceTy::callGlobalDestructors(GenericPluginTy &Plugin,
                                        DeviceImageTy &Image) {
  return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
}

Error L0DeviceTy::callGlobalCtorDtorCommon(GenericPluginTy &Plugin,
                                           DeviceImageTy &Image, bool IsCtor) {
  const char *KernelName = IsCtor ? "spirv$device$init" : "spirv$device$fini";

  // Check if a kernel was generated to run constructor or destructors.
  // It should be created by the 'spirv-lower-ctor-dtor' pass.
  GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
  if (!Handler.isSymbolInImage(*this, Image, KernelName))
    return Plugin::success();

  // Instead of returning errors directly, we capture them and provide
  // more of context about this routine.
  auto HandleErr = [&](Error Err) {
    std::string Buffer;
    llvm::raw_string_ostream(Buffer)
        << "failed to call global " << (IsCtor ? "constructors" : "destructors")
        << " in the image";
    return Plugin::error(ErrorCode::INVALID_BINARY, std::move(Err),
                         Buffer.c_str());
  };

  // The SPIR-V backend cannot handle creating the ctor / dtor array
  // automatically so we must create it ourselves. The backend will emit
  // several globals that contain function pointers we can call. These are
  // prefixed with a __init_array_object_ or __fini_array_object_.
  auto ELFObjOrErr = Handler.getELFObjectFile(Image);
  if (!ELFObjOrErr)
    return HandleErr(ELFObjOrErr.takeError());

  using FuncNameAndPriority = std::pair<StringRef, uint16_t>;
  SmallVector<FuncNameAndPriority> Funcs;
  for (ELFSymbolRef Sym : (*ELFObjOrErr)->symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr)
      return HandleErr(NameOrErr.takeError());

    if (!NameOrErr->starts_with(IsCtor ? "__init_array_object_"
                                       : "__fini_array_object_"))
      continue;

    uint16_t Priority;
    if (NameOrErr->rsplit('_').second.getAsInteger(10, Priority))
      return Plugin::error(
          ErrorCode::INVALID_BINARY,
          "failed to call global %s in the image: invalid priority",
          IsCtor ? "constructors" : "destructors");

    Funcs.emplace_back(*NameOrErr, Priority);
  }

  if (Funcs.empty()) {
    ODBG(OLDT_Module) << KernelName << " found in the image but no "
                      << (IsCtor ? "constructors" : "destructors")
                      << " found in the image.";
    return Plugin::success();
  }

  // Sort the created array to be in priority order.
  llvm::sort(Funcs,
             [](const auto &X, const auto &Y) { return X.second < Y.second; });

  auto BufferOrErr = allocate(Funcs.size() * sizeof(void *),
                              /*HostPtr=*/nullptr, TARGET_ALLOC_DEVICE);
  if (!BufferOrErr)
    return HandleErr(BufferOrErr.takeError());

  void *Buffer = *BufferOrErr;
  if (!Buffer)
    return Plugin::error(
        ErrorCode::OUT_OF_RESOURCES,
        "failed to allocate memory for global buffer to run %s",
        IsCtor ? "constructors" : "destructors");

  auto CleanupBufferAndErr = [&](Error RetErr) {
    if (auto Err = free(Buffer, TARGET_ALLOC_DEVICE)) {
      return joinErrors(std::move(RetErr), std::move(Err));
    }
    return RetErr;
  };

  auto *GlobalPtrStart = reinterpret_cast<uintptr_t *>(Buffer);
  auto *GlobalPtrStop = reinterpret_cast<uintptr_t *>(Buffer) + Funcs.size();

  SmallVector<void *> FunctionPtrs(Funcs.size());
  size_t Idx = 0;
  for (auto [Name, Priority] : Funcs) {
    GlobalTy FunctionAddr(Name.str(), sizeof(void *), &FunctionPtrs[Idx++]);
    if (auto Err = Handler.readGlobalFromDevice(*this, Image, FunctionAddr))
      return CleanupBufferAndErr(std::move(Err));
  }

  if (auto Err = dataSubmit(GlobalPtrStart, FunctionPtrs.data(),
                            FunctionPtrs.size() * sizeof(void *),
                            /*AsyncInfo=*/nullptr))
    return CleanupBufferAndErr(std::move(Err));

  GlobalTy StartGlobal(IsCtor ? "__init_array_start" : "__fini_array_start",
                       sizeof(void *), &GlobalPtrStart);
  if (auto Err = Handler.writeGlobalToDevice(*this, Image, StartGlobal))
    return CleanupBufferAndErr(std::move(Err));

  GlobalTy StopGlobal(IsCtor ? "__init_array_end" : "__fini_array_end",
                      sizeof(void *), &GlobalPtrStop);
  if (auto Err = Handler.writeGlobalToDevice(*this, Image, StopGlobal))
    return CleanupBufferAndErr(std::move(Err));

  // Call the generated kernel to execute the constructors or destructors.
  auto KernelOrErr = constructKernel(KernelName);
  if (!KernelOrErr)
    return CleanupBufferAndErr(KernelOrErr.takeError());

  GenericKernelTy &L0Kernel = *KernelOrErr;
  if (auto Err = L0Kernel.init(*this, Image))
    return CleanupBufferAndErr(std::move(Err));

  AsyncInfoWrapperTy AsyncInfoWrapper(*this, /*AsyncInfoPtr=*/nullptr);

  KernelArgsTy KernelArgs{};
  uint32_t NumBlocksAndThreads[3] = {1u, 1u, 1u};
  auto Err =
      L0Kernel.launchImpl(*this, NumBlocksAndThreads, NumBlocksAndThreads, 0,
                          KernelArgs, KernelLaunchParamsTy{}, AsyncInfoWrapper);

  AsyncInfoWrapper.finalize(Err);
  return CleanupBufferAndErr(std::move(Err));
}

} // namespace llvm::omp::target::plugin
