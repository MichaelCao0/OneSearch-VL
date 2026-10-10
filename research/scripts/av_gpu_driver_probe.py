#!/usr/bin/env python3
"""Map device-plugin UUID to CUDA via PCI identity, without unassigned compute."""
from pathlib import Path
import argparse,ctypes,datetime,json,os,re,subprocess,sys

def bus_key(text):
    m=re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',text.strip())
    if not m:raise ValueError('invalid PCI identity')
    return tuple(int(v,16) for v in m.groups())

def driver_inventory():
    cuda=ctypes.CDLL('libcuda.so.1')
    cuda.cuInit.argtypes=[ctypes.c_uint];cuda.cuInit.restype=ctypes.c_int
    result=cuda.cuInit(0)
    if result:raise RuntimeError('cuInit_error_'+str(result))
    n=ctypes.c_int();result=cuda.cuDeviceGetCount(ctypes.byref(n))
    if result:raise RuntimeError('cuDeviceGetCount_error_'+str(result))
    rows=[]
    for i in range(n.value):
        bus=ctypes.create_string_buffer(64)
        result=cuda.cuDeviceGetPCIBusId(bus,64,i)
        if result:raise RuntimeError('cuDeviceGetPCIBusId_error_'+str(result))
        rows.append({'ordinal':i,'pci_bus_id':bus.value.decode()})
    return rows

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);a.add_argument('--mask-output',type=Path,required=True);args=a.parse_args()
    if args.output.exists() or args.mask_output.exists():raise RuntimeError('refuse overwrite')
    record={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'failed','NVIDIA_VISIBLE_DEVICES':os.getenv('NVIDIA_VISIBLE_DEVICES'),'CUDA_VISIBLE_DEVICES_before':os.getenv('CUDA_VISIBLE_DEVICES'),'error_type':None,'error_code':None,'unassigned_gpu_compute':False}
    try:
        assigned=os.environ.get('NVIDIA_VISIBLE_DEVICES','')
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',assigned):raise ValueError('single allocated GPU UUID required')
        row=subprocess.check_output(['nvidia-smi','--id='+assigned,'--query-gpu=pci.bus_id,uuid','--format=csv,noheader'],text=True,timeout=30).strip().split(',')
        if len(row)!=2 or row[1].strip()!=assigned:raise ValueError('NVML assignment mismatch')
        record['assigned_pci_bus_id']=row[0].strip()
        os.environ.pop('CUDA_VISIBLE_DEVICES',None);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        rows=driver_inventory();record['driver_inventory']=rows
        matches=[r for r in rows if bus_key(r['pci_bus_id'])==bus_key(record['assigned_pci_bus_id'])]
        if len(matches)!=1:raise ValueError('allocated GPU PCI mapping is not unique')
        record['cuda_ordinal']=matches[0]['ordinal'];record['status']='mapped'
        args.mask_output.write_text(str(matches[0]['ordinal'])+'\n')
    except Exception as exc:
        record['error_type']=type(exc).__name__
        if re.fullmatch(r'cu[A-Za-z]+_error_[0-9]+',str(exc)):record['error_code']=str(exc)
    finally:
        args.output.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    return 0 if record['status']=='mapped' else 1
if __name__=='__main__':raise SystemExit(main())
