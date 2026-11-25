"""Same-H20 paired Python kernel timings from the production lowering and provider."""
import argparse
import ctypes
from hashlib import sha256
import importlib.util
import os
from functools import lru_cache
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
import torch
import triton.testing

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import load_jit_function
from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES

run = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=PROFILES, default="default")
parser.add_argument("--matrix", type=Path, required=True)
args = parser.parse_args()
profile = PROFILES[args.profile]
rows = json.loads(args.matrix.read_text())

provider_path=Path(os.environ["FLAGCV_DATA_DIR"]) / "backends/nvidia/compiler.py"
provider_spec=importlib.util.spec_from_file_location("selected_flagcv_nvidia",provider_path)
provider=importlib.util.module_from_spec(provider_spec)
provider_spec.loader.exec_module(provider)
provider.compile_kernel=lambda req, tuning, warps, grid: (tuning, warps, grid)


lib=ctypes.CDLL(os.environ["FLAGCV_IMGP_FIRST_BRIDGE"])
P=ctypes.c_void_p
I=ctypes.c_int
D=ctypes.c_double
L=ctypes.c_longlong
lib.prepare.argtypes=[I]*11+[D,D]
lib.prepare.restype=P
lib.host_bytes.argtypes=[P]
lib.host_bytes.restype=ctypes.c_size_t
lib.host_data.argtypes=[P]
lib.host_data.restype=P
lib.describe.argtypes=[P,P,L,P,I,I,I,L,P]
lib.describe.restype=ctypes.c_char_p
lib.release.argtypes=[P]
lib.last_message.restype=ctypes.c_char_p

numpy_for={"8U":np.uint8,"16U":np.uint16,"32F":np.float32}
torch_for={"8U":torch.uint8,"16U":torch.uint16,"16S":torch.int16,
           "32S":torch.int32,"32F":torch.float32}
cv_for={"8U":cv2.CV_8U,"16U":cv2.CV_16U,"32F":cv2.CV_32F}
operation={"resize":0, "warp":1, "color":2}
matrices=[(1,0,0,0,1,0),(1,0,.5,0,1,.5),(.9,-.43589,40,.43589,.9,-20)]
kernel= lru_cache(maxsize=None)(load_jit_function)


def src_shape(rows,cols,channels):
    return (rows,cols) if channels==1 else (rows,cols,channels)


def wrap(tensor, depth):
    channels=1 if tensor.ndim==2 else tensor.shape[2]
    tp=cv2.CV_MAKETYPE(cv_for[depth], channels)
    return cv2.cuda.createGpuMatFromCudaMemory(
        tensor.shape[0], tensor.shape[1], tp, tensor.data_ptr(), tensor.stride(0)*tensor.element_size())


def compare(row, input, actual, ref):
    if actual.shape != ref.shape:
        return False
    if row["depth"] != "32F":
        return np.array_equal(actual,ref)
    if row["op"]=="resize" and (row["mode"]==0 or input.shape==ref.shape):
        return np.array_equal(actual.view(np.uint32),ref.view(np.uint32))
    if row["op"]=="color" and row["code"] in (0,1,2,3,4,5,8,9):
        return np.array_equal(actual.view(np.uint32),ref.view(np.uint32))
    if not np.isfinite(actual).all():
        return False
    difference=np.abs(actual.astype(np.float64)-ref)
    if row["op"]=="resize":
        return bool(np.all(difference <= (0,4.948e-7,1.401e-6,8.419e-7)[row["mode"]]))
    if row["op"]=="warp":
        return bool(np.all(difference <= np.minimum(1e-4,5e-5*np.maximum(1.,np.abs(ref)))))
    scale=np.max(np.abs(input),axis=-1) if input.ndim==3 else np.abs(input)
    if ref.ndim==3:
        scale=scale[...,None]
    cap=np.minimum(1e-4,8*2**-24*np.maximum(np.maximum(1.,np.abs(ref)),scale))
    return bool(np.all(difference <= cap))


cv2.setNumThreads(1)
stream=torch.cuda.Stream()
native=cv2.cuda.wrapStream(stream.cuda_stream)
nulls={name:torch.empty(0,dtype=dtype,device="cuda") for name,dtype in torch_for.items()}
table_cache={}


def run_case(row):
    op=operation[row["op"]]
    sw,sh,dw,dh=(row[k] for k in ("sw","sh","dw","dh"))
    cn,mode,code,geom,border=(row[k] for k in ("channels","mode","code","geometry","border"))
    fx,fy=row["fx"],row["fy"]
    dep=row["depth"]
    a=np.empty(src_shape(sh,sw,cn),dtype=numpy_for[dep])
    hi=256 if dep=="8U" else 65536 if dep=="16U" else 2
    cv2.setRNGSeed(421)
    cv2.randu(a, ((-2 if dep=="32F" else 0),)*4, (hi,)*4)
    transform=np.array(matrices[geom],np.float64).reshape(2,3)
    if op==0:
        reference=cv2.resize(a,(dw,dh),fx=fx,fy=fy,interpolation=mode)
    elif op==1:
        w=dw if mode==cv2.INTER_CUBIC else (dw+255)//128*128
        gold=np.empty(src_shape(dh,w,cn),dtype=a.dtype)
        gold[...]=(19,23,33,18)[:cn] if cn>1 else 19
        cv2.warpAffine(a,transform,(w,dh),gold,mode,border)
        reference=gold[:,:dw]
    else:
        reference=cv2.cvtColor(a,code)

    state=lib.prepare(op,sw,sh,dw,dh,cv_for[dep],cn,mode,code,geom,border,fx,fy)
    if not state:
        raise RuntimeError(lib.last_message().decode())
    try:
        b=torch.from_numpy(a).to("cuda")
        actual=torch.empty(reference.shape,dtype=torch_for[dep],device="cuda")
        if op==1:
            actual.copy_(torch.as_tensor((19,23,33,18)[:cn] if cn>1 else 19,
                                          device="cuda",dtype=actual.dtype).expand(actual.shape))
        baseline_dst=torch.empty_like(actual)
        bytes_=lib.host_bytes(state)
        addr=lib.host_data(state) if bytes_ else 0
        storage=None
        if bytes_:
            if op==2 and addr in table_cache:
                storage=table_cache[addr]
            else:
                host=np.frombuffer(ctypes.string_at(addr,bytes_),dtype=np.uint8).copy()
                storage=torch.from_numpy(host).to("cuda")
                if op==2:
                    table_cache[addr]=storage
        target_pointer=storage.data_ptr() if storage is not None else 0
        dcn=reference.shape[2] if reference.ndim==3 else 1
        desc=lib.describe(state,b.data_ptr(),b.stride(0)*b.element_size(),
                          actual.data_ptr(),reference.shape[1],reference.shape[0],dcn,
                          actual.stride(0)*actual.element_size(),target_pointer)
        if not desc:
            raise RuntimeError(lib.last_message().decode())
        req=json.loads(desc)
        req["from_platform_registry"]="/backends/nvidia/" in req["source"]
        tuning, num_warps, dimensions=provider.compile(req)
        params={item["name"]:item["value"] for item in req["params"]}
        dims=[]
        for d in dimensions:
            if isinstance(d,int):
                dims.append(d)
                continue
            full=1
            for name,block in zip(d["product"],d.get("factor_blocks",[1]*len(d["product"]))):
                full *= triton.cdiv(params[name],block)
            dims.append(min(triton.cdiv(full,d["block"]),d.get("limit",2**63)))

        pointers=[b, actual]
        for image in req["images"][2:]:
            ptr=image["address"]
            dtype=torch_for[image["depth"]]
            if not ptr:
                pointers.append(nulls[image["depth"]])
            else:
                start=ptr-target_pointer
                width=torch.empty((),dtype=dtype).element_size()
                pointers.append(storage[start:start+width].view(dtype))
        fn=kernel(req["source"],req["function"])[tuple(dims)]
        keywords={image["name"]+"_ptr":pointer for image,pointer in zip(req["images"],pointers)}
        keywords.update({x["name"]: float(x["value"]) if x["type"]=="fp32" else x["value"]
                         for x in req["params"]})
        keywords.update(tuning, num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
        def candidate():
            return fn(**keywords)
        compiled=candidate()
        launch=dict(kernel=req["kernel"],function=compiled.metadata.name,
                    hash=compiled.hash,grid=dims,tile=tuning,num_warps=compiled.metadata.num_warps,
                    num_stages=compiled.metadata.num_stages,enable_fp_fusion=compiled.metadata.enable_fp_fusion,
                    provider_sha256=sha256(provider_path.read_bytes()).hexdigest(),
                    kernel_sha256=sha256(Path(req["source"]).read_bytes()).hexdigest())
        result=actual.cpu().numpy()
        if not compare(row,a,result,reference):
            return dict(status="FLAGCV_CPU_DIFF",max_abs=float(np.max(np.abs(result.astype(np.float64)-reference))))
        cv_src,cv_dst=wrap(b,dep),wrap(baseline_dst,dep)
        if op==0:
            baseline=lambda:cv2.cuda.resize(cv_src,(dw,dh),cv_dst,fx,fy,mode,native)
        elif op==1:
            baseline=lambda:cv2.cuda.warpAffine(cv_src,transform,(dw,dh),cv_dst,mode,border,(0.,)*4,native)
        else:
            baseline=lambda:cv2.cuda.cvtColor(cv_src,code,cv_dst,dcn,native)
        classification, diagnostic, native_difference = "COMPARABLE", None, None
        try:
            returned = baseline()
            stream.synchronize()
            if returned.cudaPtr() != baseline_dst.data_ptr():
                classification, diagnostic = "NATIVE_REFUSED", "output layout changed"
            else:
                native_result = baseline_dst.cpu().numpy()
                native_difference=float(np.max(np.abs(native_result.astype(np.float64)-reference)))
                if not compare(row, a, native_result, reference):
                    classification = "SEMANTICS_DIFFER"
        except cv2.error as error:
            classification, diagnostic = "NATIVE_REFUSED", str(error)
        pairs=[]
        pair=(("flagcv",candidate),)
        if classification != "NATIVE_REFUSED":
            pair += (("opencv_cuda",baseline),)
        for i in range(profile.samples):
            order=pair if i%2==0 else pair[::-1]
            pair_sample={"order":[name for name,_ in order]}
            for name,call in order:
                pair_sample[name]=1000*triton.testing.do_bench(
                    call,warmup=profile.warmup_ms,rep=MEASUREMENT_MS,return_mode="median")
            pairs.append(pair_sample)
        scorer=statistics.mean if args.profile=="default" else statistics.median
        flagcv_us = scorer(p["flagcv"] for p in pairs)
        native_us = scorer(p["opencv_cuda"] for p in pairs) if len(pair) == 2 else None
        return dict(classification=classification, profile=profile.name, flagcv_us=flagcv_us,
                    opencv_cuda_us=native_us, ratio=native_us/flagcv_us if classification=="COMPARABLE" else None,
                    pairs=pairs, cuda_cpu_max_abs=native_difference, launch=launch, diagnostic=diagnostic)
    finally:
        lib.release(state)


with torch.cuda.stream(stream):
    for row in rows:
        try:
            result=run_case(row)
        except Exception as error:
            result=dict(status="HARNESS_OR_RUNTIME_FAIL",error=str(error))
        print(json.dumps(dict(id=row["id"],**result)),flush=True)
