"""Check and optionally time the standard Triton border and threshold entry points."""
import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import triton

from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES
from kernels.common.copy_make_border import copy_make_border
from kernels.common.threshold import (
    threshold_apply, threshold_clear, threshold_histogram, threshold_select,
)

BORDERS = (cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT,
           cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP)
THRESHOLDS = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
BASE_MODES = range(5)
BLOCK = 256


def words(value, dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        bounds = np.iinfo(dtype)
        value = min(max(value, bounds.min), bounds.max)
    if dtype == np.float16:
        value = np.float32(value)
    bits = int.from_bytes(np.asarray(value, dtype=dtype).tobytes(), "little")
    def signed(word):
        return int.from_bytes((word & 0xFFFFFFFF).to_bytes(4, "little"), "little", signed=True)
    return signed(bits), signed(bits >> 32)


def border_call(src, dst, pad, mode, constant, dtype):
    height, width, channels = src.shape
    output_h, output_w, _ = dst.shape
    row_values = output_w * channels
    scalars = []
    for value in constant:
        if np.issubdtype(dtype, np.integer):
            value = round(value)
            if not -(2**31) <= value < 2**31:
                value = -(2**31)
        scalars.extend(words(value, dtype))
    def call():
        copy_make_border[(triton.cdiv(output_h * row_values, BLOCK),)](
            src, dst, width, height, output_h, row_values, src.stride(0), dst.stride(0),
            pad[0], pad[2], *scalars, CHANNELS=channels, MODE=mode,
            BITS=np.dtype(dtype).itemsize * 8, BLOCK=BLOCK)
    return call


def threshold_call(src, dst, cutoff, maxval, mode, dtype, work=None, auto=None, dry=False):
    kind = THRESHOLDS.index(dtype)
    rows = src.shape[0]
    row_values = src.shape[1] * src.shape[2]
    grid = (triton.cdiv(rows * row_values, BLOCK),)
    if kind < 3:
        cutoff = int(np.floor(cutoff))
        maxval = round(maxval)
        if not -(2**31) <= maxval < 2**31:
            maxval = -(2**31)
        maxval = min(max(maxval, np.iinfo(dtype).min), np.iinfo(dtype).max)
        scalars = (cutoff, 0, maxval, 0)
    else:
        scalars = (*words(cutoff, dtype), *words(maxval, dtype))
    bins = 65536 if kind == 1 else 256
    def call():
        if auto is not None:
            threshold_clear[(triton.cdiv(bins, BLOCK),)](work, bins, BLOCK=BLOCK)
            threshold_histogram[grid](src, work, rows, row_values, src.stride(0), BLOCK=BLOCK)
            threshold_select[(1,)](work, rows * row_values, BINS=bins,
                                   TRIANGLE=auto == cv2.THRESH_TRIANGLE)
        if not dry:
            threshold_apply[grid](src, dst, src if work is None else work, rows, row_values,
                                  src.stride(0), dst.stride(0), *scalars,
                                  MODE=mode, KIND=kind, AUTO=auto is not None, BLOCK=BLOCK)
    return call, cutoff


def exact(a, b):
    return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def make_pattern(dtype, shape):
    if np.issubdtype(dtype, np.floating):
        data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 3 - 17
        return data.astype(dtype)
    values = np.iinfo(dtype)
    return np.random.default_rng(820).integers(values.min, values.max + 1, shape, dtype=dtype)


def checks(device, operator=None):
    results = []
    border_depths = (np.uint8, np.int8, np.uint16, np.int16, np.int32,
                     np.float16, np.float32, np.float64)
    for dtype in border_depths if operator != "threshold" else ():
        for channels in (1, 2, 3, 4):
            backing = make_pattern(dtype, (3, 9, channels))
            host = backing[1:2, 2:4]
            src_back = torch.from_numpy(backing).to(device, copy=True)
            src = src_back[1:2, 2:4]
            pad = (4, 1, 6, 3)
            constant = (2.5, -0.5, 3.5, 65503.125)
            for mode, cv_mode in enumerate(BORDERS):
                output = torch.full((8, 15, channels), 19, dtype=src.dtype, device=device)
                roi = output[1:7, 2:13]
                expected = cv2.copyMakeBorder(host, *pad, cv_mode | cv2.BORDER_ISOLATED,
                                              value=constant).reshape(6,11,channels)
                border_call(src, roi, pad, mode, constant, dtype)()
                ok = exact(roi.cpu().numpy(), expected)
                roi.fill_(19)
                guarded = bool((output == 19).all().item()) and exact(src_back.cpu().numpy(), backing)
                results.append({"op":"copyMakeBorder", "depth":np.dtype(dtype).name,
                                "channels":channels,"mode":mode,"pass":ok and guarded})
    if operator == "copyMakeBorder":
        return results
    for dtype in THRESHOLDS:
        for mode in BASE_MODES:
            limits = (-32769, -32768, -32767.25, 32767, 32768) if dtype == np.int16 and mode == 2 else (2.5,)
            for cutoff in limits:
                host = make_pattern(dtype, (2,5,1))
                if dtype == np.int16 and mode == 2:
                    host[0,:3,0] = [-32768,-7,9]
                if dtype in (np.float32,np.float64):
                    host[0,:,0] = [np.nan,-0.,0.,5.,np.nan]
                    host[1,:,0] = [np.inf,-np.inf,-0.,0.,cutoff]
                elif not (dtype == np.int16 and mode == 2):
                    host[1,:3,0] = [1,2,3]
                src = torch.from_numpy(host).to(device)
                dst = torch.empty_like(src)
                call, got_ret = threshold_call(src,dst,cutoff,7.5,mode,dtype)
                ret, ref = cv2.threshold(host,cutoff,7.5,mode)
                call()
                results.append({"op":"threshold","depth":np.dtype(dtype).name,"mode":mode,
                                "cutoff":cutoff,"pass":ret==got_ret and exact(dst.cpu().numpy(),ref)})
    auto_cases = ((np.uint8, cv2.THRESH_OTSU, [0,1,1,1,4,65,65,255]),
                  (np.uint8, cv2.THRESH_OTSU, [0,16,32]),
                  (np.uint8, cv2.THRESH_OTSU, [0]*8 + [128]*9 + [255]*8),
                  (np.uint8, cv2.THRESH_TRIANGLE, [0,1,1]),
                  (np.uint16, cv2.THRESH_OTSU, [0,1,1,1,4000,65000,65001,65535]))
    for dtype, method, values in auto_cases:
        host = np.array(values, dtype=dtype).reshape(1,-1,1)
        src = torch.from_numpy(host).to(device)
        bins = 65536 if dtype==np.uint16 else 256
        scratch = torch.empty(bins+1,dtype=torch.int32,device=device)
        # Interpreting the ordered 65,536-bin recurrence is expensive.
        # The device suite still checks all five modes and dry-run.
        interpreted_wide = device == "cpu" and dtype == np.uint16
        modes = (cv2.THRESH_BINARY,) if interpreted_wide else BASE_MODES
        for mode in modes:
            dst = torch.full_like(src,19)
            call, _ = threshold_call(src,dst,0,42,mode,dtype,scratch,method)
            ret, ref = cv2.threshold(host,0,42,mode|method)
            call()
            good = ret==scratch[0].item() and exact(dst.cpu().numpy(),ref)
            if not interpreted_wide:
                dst.fill_(19)
                dry, _ = threshold_call(src,dst,0,42,mode,dtype,scratch,method,dry=True)
                dry()
                good = good and ret==scratch[0].item() and bool((dst==19).all().item())
            results.append({"op":"threshold","depth":np.dtype(dtype).name,"mode":mode,
                            "automatic":method,"cutoff":ret,"pass":good})
    return results


def time_profiles(device, name, operator):
    from triton.testing import do_bench
    protocol = PROFILES[name]
    rows = []
    for h,w in ((64,127),(1080,1920),(2160,3840)):
        source = torch.zeros((h,w,1),dtype=torch.uint8,device=device)
        border_output = torch.empty((h+6,w+6,1),dtype=source.dtype,device=device)
        call_border = border_call(source,border_output,(3,3,3,3),2,(0,0,0,0),np.uint8)
        output = torch.empty_like(source)
        call_threshold,_ = threshold_call(source,output,126,255,0,np.uint8)
        for name_op,call in (("copyMakeBorder",call_border),("threshold",call_threshold)):
            if operator and name_op != operator:
                continue
            samples = [do_bench(call,warmup=protocol.warmup_ms,rep=MEASUREMENT_MS,
                                return_mode="median") for _ in range(protocol.samples)]
            reduce = mean if name=="default" else median
            rows.append({"op":name_op,"source":[h,w],"samples_ms":samples,"latency_ms":reduce(samples),
                         "metric":"triton device event","baseline":None})
    return {"protocol":protocol._asdict(),"measurement_ms":MEASUREMENT_MS,"rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(PROFILES))
    parser.add_argument("--operator", choices=("copyMakeBorder", "threshold"))
    args=parser.parse_args()
    if args.benchmark and args.device=="cpu":
        parser.error("benchmarking requires an admitted accelerator")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evidence=checks(args.device,args.operator)
    failed=[case for case in evidence if not case["pass"]]
    result={"device":args.device,"opencv_version":cv2.__version__,"cases":evidence}
    if not failed and args.benchmark:
        result["performance"]=time_profiles(args.device,args.benchmark,args.operator)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for case in failed:
        print(case)
    print(len(evidence),"fixtures",len(failed),"failed")
    return bool(failed)


if __name__=="__main__":
    sys.exit(main())
