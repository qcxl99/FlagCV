"""Host-only scratch ABI checks. Triton is only needed for real codegen."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compiler"))
from flagcv_codegen.main import load_provider  # noqa: E402
from flagcv_codegen.scratch_abi import (  # noqa: E402
    validated_hip_scratch_args,
    validated_scratch_args,
)

ROOT = Path(__file__).resolve().parents[2]


def provider(name):
    return load_provider(name, ROOT)


def verify_compiler(name, backend, version, arch=None, module=None, wave=None):
    target = SimpleNamespace(backend=backend, arch=arch, warp_size=wave)
    provider(name).qualify(target, version, module)


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


class VendorPlatformFacts(unittest.TestCase):
    def test_qualified_hip_targets_retain_facts_and_launch_plan(self):
        for backend, target, warp in (
            ("amd", "hip:gfx1100:32", 32),
            ("hygon", "hip:gfx936:64", 64),
            ("hygon", "hip:gfx938:64", 64),
        ):
            with self.subTest(target=target):
                compiler = provider(backend)
                tuning, warps, grid = compiler.kernel_config(dict(
                    backend=backend, target=target, kernel="flip", plan="flip",
                    images=[{"depth": "32F"}], params=[]))
                self.assertEqual(compiler.FACTS, dict(triton_backend="hip", binary="hsaco",
                                             extension="hsaco", warp=warp, tile=2048,
                                             max_warps=16))
                self.assertEqual(tuning, dict(BLOCK_X=512, BLOCK_Y=4, ROW_GRID_LIMIT=0))
                self.assertEqual(warps, 4)
                self.assertEqual(grid, [{"product": ["width", "CHANNELS"], "block": 512},
                                        {"product": ["height"], "block": 4}, 1])

    def test_vendor_tuning_isolated_in_both_directions(self):
        self.assertIsNot(provider("amd").FACTS, provider("hygon").FACTS)
        for first, second in ((("amd", "hip:gfx1100:32"), ("hygon", "hip:gfx936:64")),
                              (("hygon", "hip:gfx938:64"), ("amd", "hip:gfx1100:32"))):
            with self.subTest(edited=first[0]), patch.dict(provider(first[0]).FACTS, tile=256, max_warps=2):
                def plan(selection):
                    backend, target = selection
                    return provider(backend).kernel_config(dict(backend=backend, target=target, kernel="flip", plan="flip",
                                                  images=[{"depth": "8U"}], params=[]))

                edited, edited_warps, _ = plan(first)
                untouched, untouched_warps, _ = plan(second)
                self.assertEqual((edited["BLOCK_X"], edited_warps), (256, 2))
                self.assertEqual((untouched["BLOCK_X"], untouched_warps), (2048, 4))

    def test_cross_vendor_and_incorrect_wave_remain_unsupported(self):
        for backend, target in (("amd", "hip:gfx936:64"), ("amd", "hip:gfx938:64"),
                                ("hygon", "hip:gfx1100:32"), ("hygon", "hip:gfx936:32")):
            with self.subTest(backend=backend, target=target):
                with self.assertRaisesRegex(ValueError, "unsupported target"):
                    provider(backend).kernel_config(dict(target=target))


class ScratchABI(unittest.TestCase):
    def test_nvidia_rejects_each_ineligible_value_before_toolchain_lookup(self):
        module = "triton.backends.nvidia.compiler"
        with patch.dict(sys.modules, {"triton": None}), patch(
                "importlib.metadata.distribution", side_effect=AssertionError("looked up FlagTree")):
            for backend, version, compiler, wave in (
                ("hip", "3.6.0", module, 32), ("cuda", "3.6.0", module, 64),
                ("cuda", "3.7.0", module, 32), ("cuda", "3.6.0", "custom.compiler", 32),
            ):
                with self.subTest(identity=(backend, version, compiler, wave)):
                    with self.assertRaisesRegex(RuntimeError, "unqualified"):
                        verify_compiler("nvidia", backend, version, 90, compiler, wave)

    def test_nvidia_qualifies_exact_distribution_and_import_path(self):
        from importlib import metadata

        compiler = provider("nvidia")
        target = SimpleNamespace(backend="cuda", arch=90, warp_size=32)
        module = "triton.backends.nvidia.compiler"
        installed = Path("/qualified/triton/__init__.py")
        distribution = SimpleNamespace(version=compiler.FLAGTREE_RELEASE,
                                       locate_file=lambda path: installed.parent.parent / path)
        with patch("importlib.metadata.distribution", side_effect=metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "unqualified"):
                compiler.qualify(target, "3.6.0", module)
        with patch("importlib.metadata.distribution", return_value=distribution) as lookup:
            with patch.dict(sys.modules, {"triton": SimpleNamespace(__file__=str(installed))}):
                compiler.qualify(target, "3.6.0", module)
                lookup.assert_called_once_with("flagtree-nvidia")
                distribution.version = "other"
                with self.assertRaisesRegex(RuntimeError, "unqualified"):
                    compiler.qualify(target, "3.6.0", module)
            distribution.version = compiler.FLAGTREE_RELEASE
            with patch.dict(sys.modules, {"triton": SimpleNamespace(__file__="/other/triton.py")}):
                with self.assertRaisesRegex(RuntimeError, "unqualified"):
                    compiler.qualify(target, "3.6.0", module)

    def test_verified_layout(self):
        for arch in ("gfx936", "gfx938"):
            verify_compiler("hygon", "hip", "3.6.0", arch,
                                       "triton.backends.hcu.compiler_hcu", 64)
        verify_compiler("amd", "hip", "3.6.0", "gfx1100",
                                   "triton.backends.amd.compiler", 32)
        self.assertEqual(validated_scratch_args(ZERO, PTX, SIGNATURE), 2)
        compiled = SimpleNamespace(metadata=ZERO, asm={"ptx": PTX})
        self.assertEqual(provider("nvidia").validate_scratch(compiled, SIGNATURE), 2)

    def test_unqualified_compiler_or_platform(self):
        for identity in (("nvidia", "cuda", "3.7.0"), ("hygon", "hip", "3.6.0"),
                         ("ascend", "npu", "3.6.0"), ("mthreads", "mtgpu", "3.6.0"),
                         ("iluvatar", "ix", "3.6.0"), ("thead", "PPU", "3.6.0")):
            with self.subTest(identity=identity), self.assertRaisesRegex(RuntimeError, "unqualified"):
                verify_compiler(*identity)

    def test_vendor_target_wave_and_compiler_must_match(self):
        amd = "triton.backends.amd.compiler"
        hcu = "triton.backends.hcu.compiler_hcu"
        for vendor, arch, module, wave in (
            ("amd", "gfx936", hcu, 64), ("amd", "gfx938", hcu, 64),
            ("hygon", "gfx1100", amd, 32), ("hygon", "gfx936", hcu, 32),
            ("hygon", "gfx938", amd, 64), ("amd", "gfx1100", hcu, 32),
            ("amd", "gfx942", amd, 64),
        ):
            with self.subTest(vendor=vendor, arch=arch, wave=wave):
                with self.assertRaisesRegex(RuntimeError, "unqualified"):
                    verify_compiler(vendor, "hip", "3.6.0", arch, module, wave)

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
