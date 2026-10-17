import importlib.util
import json
import platform
import sys

modules=['fastapi','uvicorn','cv2','numpy','sklearn','PIL','multipart','pytest']
report={"python":sys.version.split()[0],"os":platform.system(),"modules":{m:importlib.util.find_spec(m) is not None for m in modules}}
if importlib.util.find_spec('torch'):
    import torch
    report.update(cuda=torch.cuda.is_available())
    if torch.cuda.is_available():report.update(gpu=torch.cuda.get_device_name(0),vram_gb=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))
print(json.dumps(report,indent=2))
