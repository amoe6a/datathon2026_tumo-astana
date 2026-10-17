"""Run during setup only. Runtime inference sets offline mode and never downloads."""
import argparse
from huggingface_hub import snapshot_download

p=argparse.ArgumentParser()
p.add_argument("--output",default="models/qwen3-vl-2b")
p.add_argument("--revision",default="78448d793a7eb2f7a987a1da76d464384aa1becd")
a=p.parse_args()
snapshot_download("Qwen/Qwen3-VL-2B-Instruct",revision=a.revision,local_dir=a.output,
                  allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","LICENSE","README.md"])
print("Weights downloaded. Runtime uses local_files_only=True; retain LICENSE with the model.")
