#!/usr/bin/env python3
"""Existing operator test runner, now exercises all floating dtypes on CPU or NVIDIA."""
import argparse, importlib, torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',choices=['cpu','nvidia'],default='nvidia')
args=p.parse_args();torch.manual_seed(20260919)
cases={'add':[((3,129),)],'argmax':[((4097,),)],'embedding':[((7,),(37,65))],
       'linear':[((3,19),(3,33),(19,33),True),((16,128),(16,128),(128,128),False)],
       'rms_norm':[((5,129),)],'rope':[((5,4,128),(7,12))],
       'swiglu':[((5,129),)],'self_attention':[(5,11,4,2,128),(1,17,8,2,40)]}
count=0
for op,shapes in cases.items():
 fn=getattr(importlib.import_module('ops.'+op),'test_op_'+op)
 for dtype,tol in [('f32',2e-5),('f16',.003),('bf16',.03)]:
  for shape in shapes:
   kw=dict(dtype_name=dtype,device_name=args.device)
   if op not in ['argmax','embedding']:kw.update(atol=tol,rtol=tol)
   fn(*shape,**kw);count+=1
print(f'PASS {count} operator cases, device={args.device}, dtypes=F32/F16/BF16')
