# BC-250 Hardware Notes

## APU Specifications
- **Architecture**: Zen 2 CPU (8 cores / 16 threads) + semi-custom RDNA 1.5 GPU (Cyan Skillfish / Oberon, `gfx1013`). Combines RDNA 2 compute unit layout, high clock targets, and Ray Tracing BVH units with an RDNA 1-style memory subsystem (no Infinity Cache / System Level Cache) and packed dual-rate FP16.
- **Compute Units**: 40 CUs (20 WGPs, physically present on the PS5-derived die).
  - *Stock Mining Board State*: Often ships software-limited to 24 CUs (12 WGPs).
  - *40 CU Unlock*: Re-enabled via the community `amdgpu` kernel patch ([duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock)) or distributions like Bazzite/SkillFishOS.
- **Hardware Video Block**: VCN 2.0.3 (Video Core Next, `UVD_VERSION = 0x0002001B`) — silicon physically present on die, but permanently unprovisioned, isolated behind the Data Fabric Access Control List (`SEC_GASKET~0x24`), and unmanaged by SMU/VBIOS.
- **Codename**: Cyan Skillfish (Device ID: `1002:13fe`)

## Memory Map & Architecture
- **Unified GDDR6 Pool**: All 16 GB is a single unified pool of high-bandwidth GDDR6 shared by CPU and GPU.
- **VRAM vs GTT**: The `512 MB` reported in `mem_info_vram_total` is merely a kernel-level label for the initial aperture slice; the GPU accesses the unified memory via GART/GTT (`amdgpu.gttsize`). Vulkan RADV exposes ~7.95 GiB across two heaps. Attempting to force larger "dedicated VRAM" in BIOS/APCB is ineffective and unnecessary.

## Known Hardware Realities

### 1. Physical VCN Cannot Be Unlocked (Architectural Proof)
Exhaustive reverse-engineering across the community and firmware audit of the AMD Platform Security Processor (PSP) and System Management Unit (SMU) confirmed that hardware VCN cannot be enabled:
- **Data Fabric Access Control List (`SEC_GASKET~0x24`)**: At early boot, the encrypted PSP bootloader (`PSP_BL`) executes a signed table of 926 register writes. ~816 of these writes program the Data Fabric (DF) access-control registers (`0x09xxxxxx`), locking all non-PSP masters (Host CPU `0xB8/0xBC`, SMU mailbox `sec_smn_write32`, GPU `regs_pcie`) out of the VCN aperture. It permanently latches `CC_UVD_HARVESTING = 0x3` (at SMN `0x1f81c`) and `[0x1f820] = 0x00185103`.
- **Hazardous Register Warning (`0x1f81c`)**: Directly reading SMN register `0x1f81c` from host space triggers a hardware fabric hang that wedges the APU and requires an AC power cycle.
- **Cryptographic Sealing**: Only the PSP itself (via `svc #0x7c`) can bypass the fabric ACL, but all 7 investigated exploit vectors (including APCB parsing in ABL4) are definitively closed on Cyan Skillfish. Modifying the `$KDB` key database in SPI flash permanently bricks the motherboard.
- **Upstream AMD Confirmation**: AMD Linux kernel maintainer Alex Deucher confirmed that VCN was never part of the BC-250 product definition. The SMU PMFW contains zero VCN power/clock handlers, the VBIOS contains zero VCN tables, and no signed `vcn_2_0_3.bin` firmware exists for this SKU.
- **Conclusion**: Physical VCN cannot be revived by any BIOS mod or kernel patch. This driver (`bc250-encoding-decoding-fix`) is the sole viable solution, executing encode operations as custom Vulkan Compute shaders across the 40 Compute Units.

### 2. DisplayPort / HDMI Audio Clock Divisor
- The display controller (`dc`) calculates an incorrect audio sample clock divisor for 44.1/48 kHz audio. The included `bc250_audio_fix` DKMS module writes the proper clock ratios directly to APU DCCG registers (`0x05E0`, `0x05E4`, `0x05E8`).

<!-- bc250-encoding-decoding-fix v0.4.3 -->
