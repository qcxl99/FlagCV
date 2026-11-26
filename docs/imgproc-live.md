# Installed Imgproc live benchmark

Build the existing live runners with `FLAGCV_BENCHMARK_IMGPROC=ON`. The
runner backend defaults to NVIDIA when its plugin is built and otherwise HIP.
Set `FLAGCV_BENCHMARK_IMGPROC_BACKEND=hip` or `hygon` to select a built HIP
plugin explicitly. HIP runners require the HIP SDK and OpenCV core and
imgproc. The existing NVIDIA runners retain their OpenCV CUDA comparisons.
The core FlagCV runtime keeps its plugin linkage.

After installing or moving the prefix, use the installed checker.

```sh
python /path/to/prefix/share/flagcv/benchmark/check_nvidia.py \
    --imgproc-live --prefix /path/to/prefix --backend hip \
    --device-uuid physical-device-uuid --verify-only --summary verify.json
```

Bind the runtime to one visible device before launching. Keep the device idle
and use the site's physical-card lock for measurements. `--verify-only` runs
the candidate against its CPU oracle and checks guards without collecting
timing samples or running a comparator. Repeat `--case` to select case IDs.
The default list is representative. `--case-manifest` also accepts the
existing first-family archive manifest or the KernelForge legacy mapped
manifest. The latter generates and hashes one input shared across the two
processes and keeps the original transparent background. The 48 installed
batch cases are selected through their ordinary IDs.
When several historical IDs name one parameter group, the checker runs the
group once and records every explicitly requested ID in
`requested_case_groups`. The mapping retains the selected row weights.

For HIP, the C++ runner and Python must use the same OpenCV core and imgproc
libraries and one resolved HIP runtime library. Select an environment in which
the plugin, the runner and Torch use that runtime.

HIP event samples use warm buffers without clearing L2. The default
profile averages two round medians after 100 ms warmup and a 100 ms
measurement window per round. The long profile uses the median of three
rounds with 1000 ms warmup and a 100 ms measurement window. C++ HIP
collects event and synchronous wall samples on separate submissions
within its measurement window. The report distinguishes the compiler's
C++ grid plan from Python's evaluated launch grid.

The optional `--baseline-adapter` file supplies `NAME` and
`prepare_first(case, source, destination, *, transform)` for the Python
first-family HIP runs. Return a stream-ordered callable that writes the
provided destination, or return `None` for an unavailable equivalent.
The runner checks the comparator output against the same CPU reference
before enabling a ratio. Select an adapter compatible with the candidate
process's loaded libraries. Keep measurements that need different SDK or
OpenCV builds in isolated processes. Other unpaired rows retain candidate
times and a null ratio. The C++ HIP runner has no required ATen or Torch
dependency.
