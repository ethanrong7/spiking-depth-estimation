"""CPU construction/BN/reset checks only; no forward/native memory claim."""
import json
from pathlib import Path
import torch
from models import build_model,reset_model


def main():
 torch.set_num_threads(1);rows=[]
 for name in ['original','raft_stereo','selective_raft']:
  model=build_model(name,1234);model.train();model.freeze_bn()
  assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
  memories=[m for m in model.modules() if hasattr(m,'_memories') and 'v' in m._memories]
  for m in memories:m.v=torch.ones(1);m.spike=torch.ones(1)
  reset_model(model)
  assert all(isinstance(m.v,(float,int)) and m.v==0 for m in memories)
  rows.append(dict(model=name,parameters=sum(p.numel() for p in model.parameters()),fnet_input_channels=model.fnet.conv1.in_channels,cnet_input_channels=model.cnet.conv1.in_channels,frozen_BN=True,reset_membrane_sites=len(memories),reset_restores_registered_zero=True,scope='CPUconstructiononly,no_model_forward'))
  del model
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
