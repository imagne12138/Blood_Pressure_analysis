"""Quick verification of fusion model and dataset imports."""
import sys
sys.path.insert(0, 'e:/Kaggle_projects/Blood_Pressure_analysis')

# 1. Verify model
from model.Fusion_Model_26 import FusionModel26
import torch
m = FusionModel26(filters=[1,32,64,128], num_layers=2)
print(f'Model params: {sum(p.numel() for p in m.parameters()):,}')
x = torch.randn(4, 1, 1024)
f = torch.randn(4, 26)
o = m(x, f)
print(f'Input PPG:  {x.shape}')
print(f'Input Feat: {f.shape}')
print(f'Output:     {o.shape}')

# 2. Verify dataset
from utils.create_data import Fusion26Dataset, LoadFusion26Dataset
print('Fusion26Dataset imported OK')

print('All checks passed!')
