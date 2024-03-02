import os 
import json
import torch 
import wandb
import safetensors
from pathlib import Path 
# 파일이 속한 폴더의 path의 directory를 만듬
def create_necessary_folder(path):
  path = '/'.join(path.split('/')[:-1])
  Path(path).mkdir(parents=True,exist_ok=True)

def safe_save(ckpt,path): #checkpoint
  try:
    os.remove(f'{path}.bak')  : #path.bak라는 이름의 백업 파일이 존재하면 삭제합니다.
  except OSError: 
    pass
  try: 
    os.rename(path,f'{path}.bak') 
  except OSError:
    pass
  if path.endswith('.pt') or path.endswith('.ckpt'):
    torch.save(ckpt,path) #pt,ckpt면 ckpt를 path에다가 저장
  elif path.endswith('.json'):
    with open(path,'w',encoding='utf-8') as f:
      json.dump(ckpt,f,indent=4)
  elif path.endswith(".safetensors"):
        safetensors.torch.save_file(ckpt, path)
  else:
        raise ValueError(f"File extension not supported: {path}")

  
    
