"""Host-only scratch ABI checks. Triton is only needed for real codegen."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compiler"))
from flagcv_codegen.scratch_abi import (  # noqa: E402
    require_qualified_compiler,
    validated_hip_scratch_args,
    validated_scratch_args,
)

SIGNATURE = {"src_ptr": "*u8", "width": "i32", "BLOCK": "constexpr"}
PTX = """
.visible .entry sample(
  .param .u64 .ptr .global .align 1 sample_param_0,
  .param .u32 sample_param_1,
  .param .u64 .ptr .global .align 1 sample_param_2,
  .param .u64 .ptr .global .align 1 sample_param_3
)
"""
ZERO = SimpleNamespace(name="sample", global_scratch_size=0, profile_scratch_size=0)


class ScratchABI(unittest.TestCase):
    def test_verified_layout(self):
        require_qualified_compiler("nvidia", "cuda", "3.6.0")
        for arch in ("gfx936", "gfx938"):
            require_qualified_compiler("hygon", "hip", "3.6.0", arch,
                                       "triton.backends.hcu.compiler_hcu", 64)
        require_qualified_compiler("amd", "hip", "3.6.0", "gfx1100",
                                   "triton.backends.amd.compiler", 32)
        self.assertEqual(validated_scratch_args(ZERO, PTX, SIGNATURE), 2)

    def test_unqualified_compiler_or_platform(self):
        for identity in (("nvidia", "cuda", "3.7.0"), ("hygon", "hip", "3.6.0"),
                         ("ascend", "npu", "3.6.0"), ("mthreads", "mtgpu", "3.6.0"),
                         ("iluvatar", "ix", "3.6.0"), ("thead", "PPU", "3.6.0")):
            with self.subTest(identity=identity), self.assertRaisesRegex(RuntimeError, "unqualified"):
                require_qualified_compiler(*identity)

    def test_vendor_target_wave_and_compiler_must_match(self):
        amd = "triton.backends.amd.compiler"
        hcu = "triton.backends.hcu.compiler_hcu"
        for vendor, arch, module, wave in (
            ("amd", "gfx936", hcu, 64), ("amd", "gfx938", hcu, 64),
            ("hygon", "gfx1100", amd, 32), ("hygon", "gfx936", hcu, 32),
            ("hygon", "gfx938", amd, 64), ("amd", "gfx1100", hcu, 32),
            ("hip", "gfx936", hcu, 64), ("amd", "gfx942", amd, 64),
        ):
            with self.subTest(vendor=vendor, arch=arch, wave=wave):
                with self.assertRaisesRegex(RuntimeError, "unqualified"):
                    require_qualified_compiler(vendor, "hip", "3.6.0", arch, module, wave)

    def test_requires_both_zero_metadata_fields(self):
        for metadata in (SimpleNamespace(name="sample", global_scratch_size=0),
                         SimpleNamespace(name="sample", profile_scratch_size=0),
                         SimpleNamespace(name="sample", global_scratch_size=0, profile_scratch_size=32)):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(RuntimeError, "scratch"):
                validated_scratch_args(metadata, PTX, SIGNATURE)

    def test_hip_pointer_layout_and_absent_global_metadata(self):
        metadata = SimpleNamespace(name="sample", profile_scratch_size=0)
        entry = ("define amdgpu_kernel void @sample(ptr addrspace(1) inreg %0, "
                 "i32 inreg %1, ptr addrspace(1) inreg readnone %2, "
                 "ptr addrspace(1) inreg readnone %3) #0 {")
        self.assertEqual(validated_hip_scratch_args(metadata, entry, SIGNATURE), 2)
        with self.assertRaisesRegex(RuntimeError, "active scratch"):
            validated_hip_scratch_args(metadata, entry.replace("readnone %2", "readonly %2"),
                                       SIGNATURE)
        metadata.global_scratch_size = 4
        with self.assertRaisesRegex(RuntimeError, "needs scratch"):
            validated_hip_scratch_args(metadata, entry, SIGNATURE)

    def test_checks_emitted_entry(self):
        samples = (None, PTX.replace("sample_param_3", "sample_param_4"),
                   PTX.replace(".u32 sample_param_1", ".u64 sample_param_1"),
                   PTX.replace(".u64 .ptr .global .align 1 sample_param_3", ".u64 sample_param_3"))
        for ptx in samples:
            with self.subTest(ptx=ptx), self.assertRaisesRegex(RuntimeError, "entry parameter ABI"):
                validated_scratch_args(ZERO, ptx, SIGNATURE)


if __name__ == "__main__":
    unittest.main()
