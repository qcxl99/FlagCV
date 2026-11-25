import argparse
import ctypes as c
from pathlib import Path
from functools import lru_cache
import importlib.util
import json
import os
import statistics

import cv2
import numpy as np
import torch
import triton.testing

from flagcv_codegen.triton_build import load_jit_function
from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES

run=Path(__file__).resolve().parent
lib=c.CDLL(os.environ["FLAGCV_IMGP_BATCH_BRIDGE"])
P,I,L=c.c_void_p,c.c_int,c.c_longlong
lib.prepare.argtypes=[I]*11+[c.c_double,c.c_double]
lib.prepare.restype=P
lib.host_bytes.argtypes=[P]
lib.host_bytes.restype=c.c_size_t
lib.host_data.argtypes=[P]
lib.host_data.restype=P
lib.describe.argtypes=[P,P,L,P,I,I,I,L,P,I,L,L]
lib.describe.restype=c.c_char_p
lib.release.argtypes=[P]
lib.last_message.restype=c.c_char_p
kernel=lru_cache(maxsize=None)(load_jit_function)
parser = argparse.ArgumentParser(description="Time installed 8U resizeBatch kernels")
parser.add_argument("--matrix", type=Path, required=True)
parser.add_argument("--profile", choices=PROFILES, default="default")
args = parser.parse_args()
profile = PROFILES[args.profile]
spec=importlib.util.spec_from_file_location(
    "h20_nvidia_batch_provider", Path(os.environ["FLAGCV_DATA_DIR"])/"backends/nvidia/compiler.py")
provider=importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)
provider.compile_kernel=lambda req,tuning,warps,grid:(tuning,warps,grid)

stream=torch.cuda.Stream()
native=cv2.cuda.wrapStream(stream.cuda_stream)
cv2.setNumThreads(1)

def wrap(mat, channels):
    return cv2.cuda.createGpuMatFromCudaMemory(mat.shape[0],mat.shape[1],
        cv2.CV_MAKETYPE(cv2.CV_8U,channels),mat.data_ptr(),mat.stride(0))


def run_case(row):
    sw,sh,dw,dh=(row[k] for k in ("sw","sh","dw","dh"))
    fx,fy=row["fx"],row["fy"]
    n,cn,mode=(row[k] for k in ("count","channels","mode"))
    host=np.random.default_rng(809).integers(0,256,(n,sh,sw,cn),dtype=np.uint8)
    gold=[cv2.resize(a,(dw,dh),fx=fx,fy=fy,interpolation=mode) for a in host]
    expected=np.stack([a.reshape(a.shape[0],a.shape[1],cn) for a in gold])
    height,width=expected.shape[1:3]
    source=torch.empty((n,sh+3,sw+5,cn),dtype=torch.uint8,device="cuda")[:,1:sh+1,1:sw+1]
    source.copy_(torch.from_numpy(host))
    dest=torch.empty((n,height+5,width+7,cn),dtype=torch.uint8,device="cuda")[:,2:height+2,2:width+2]
    other=torch.empty((n,height+5,width+7,cn),dtype=torch.uint8,device="cuda")[:,2:height+2,2:width+2]
    state=lib.prepare(0,sw,sh,dw,dh,0,cn,mode,-1,0,0,fx,fy)
    if not state:
        raise RuntimeError(lib.last_message().decode())
    try:
        size=lib.host_bytes(state)
        table=None
        if size:
            data=np.frombuffer(c.string_at(lib.host_data(state),size),dtype=np.uint8).copy()
            table=torch.from_numpy(data).to("cuda")
        addr=table.data_ptr() if table is not None else 0
        raw=lib.describe(state,source.data_ptr(),source.stride(1),dest.data_ptr(),
                         width,height,cn,dest.stride(1),addr,n,source.stride(0),dest.stride(0))
        if not raw:
            raise RuntimeError(lib.last_message().decode())
        req=json.loads(raw)
        req["from_platform_registry"]=False
        tune,warps,dimensions=provider.compile(req)
        params={x["name"]:x["value"] for x in req["params"]}
        grid=[]
        for d in dimensions:
            if isinstance(d,int):
                grid.append(d)
            else:
                value=1
                for k,g in zip(d["product"],d.get("factor_blocks",[1]*len(d["product"]))):
                    value*=triton.cdiv(params[k],g)
                grid.append(min(triton.cdiv(value,d["block"]),d.get("limit",2**63)))
        kw={"src_ptr":source,"dst_ptr":dest}
        for p in req["images"][2:]:
            off=p["address"]-addr
            typ=torch.float32 if p["depth"]=="32F" else torch.int32
            kw[p["name"]+"_ptr"]=table[off:off+4].view(typ)
        kw.update(tune,num_warps=warps,num_stages=1,enable_fp_fusion=False)
        kw.update({p["name"]:float(p["value"]) if p["type"]=="fp32" else p["value"]
                   for p in req["params"]})
        dispatch=kernel(req["source"],req["function"])[tuple(grid)]
        def candidate():
            return dispatch(**kw)
        compiled = candidate()
        if not np.array_equal(dest.cpu().numpy(),expected):
            return dict(status="FLAGCV_CPU_DIFF")
        views=[(wrap(source[i],cn),wrap(other[i],cn)) for i in range(n)]
        def baseline():
            for s,d in views:
                cv2.cuda.resize(s,(dw,dh),d,fx,fy,mode,native)
        native_delta, diagnostic = None, None
        try:
            baseline()
            stream.synchronize()
            if any(d.cudaPtr()!=other[i].data_ptr() for i, (_,d) in enumerate(views)):
                diagnostic = "native reallocated a destination"
            else:
                native_delta=int(np.max(np.abs(other.cpu().numpy().astype(np.int32)-expected)))
        except cv2.error as error:
            diagnostic = str(error)
        pairs=[]
        for i in range(profile.samples):
            loops=(("flagcv",candidate),)
            if diagnostic is None:
                loops += (("opencv_cuda",baseline),)
            if i%2: loops=loops[::-1]
            pair={}
            for name,f in loops:
                pair[name]=1000*triton.testing.do_bench(f,warmup=profile.warmup_ms,
                       rep=MEASUREMENT_MS,return_mode="median")
            pairs.append(pair)
        scorer = statistics.mean if args.profile=="default" else statistics.median
        flag = scorer(p["flagcv"] for p in pairs)
        native_us = scorer(p["opencv_cuda"] for p in pairs) if diagnostic is None else None
        state = "NATIVE_REFUSED" if diagnostic else "COMPARABLE" if native_delta == 0 else "SEMANTICS_DIFFER"
        return dict(classification=state, pairs=pairs, flagcv_us=flag, opencv_cuda_us=native_us,
                    ratio=native_us/flag if state=="COMPARABLE" else None,
                    launch=dict(kernel=req["kernel"],grid=grid,tile=tune,hash=compiled.hash,
                                num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                                enable_fp_fusion=compiled.metadata.enable_fp_fusion),
                    cuda_cpu_max_abs=native_delta,flagcv_launches=1,
                    opencv_cuda_launches=n if diagnostic is None else None,diagnostic=diagnostic)
    finally:
        lib.release(state)

with torch.cuda.stream(stream):
    for row in json.loads(args.matrix.read_text()):
        try:
            result=run_case(row)
        except Exception as e:
            result=dict(status="HARNESS_OR_RUNTIME_FAIL",error=str(e))
        print(json.dumps(dict(id=row["id"],**result)),flush=True)
