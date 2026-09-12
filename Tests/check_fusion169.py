"""Quick verify Fusion169 model + dataset."""
import sys
sys.path.insert(0, 'e:/Kaggle_projects/Blood_Pressure_analysis')

from model.Fusion_Model_169 import FusionModel169
import torch

m = FusionModel169(filters=[1,32,64,128], num_layers=2, proj_dim=64)
print(f'Model params: {sum(p.numel() for p in m.parameters()):,}')
x = torch.randn(4, 1, 1024)
f = torch.randn(4, 169)
o = m(x, f)
print(f'Input PPG:  {x.shape}')
print(f'Input Feat: {f.shape}')
print(f'Output:     {o.shape}')

# Check fusion dim is 320
for name, p in m.named_parameters():
    if 'linear_sbp_out.0' in name:
        print(f'SBP head first layer: {p.shape} (expect [128, 320])')
    if 'linear_dbp_out.0' in name:
        print(f'DBP head first layer: {p.shape} (expect [128, 320])')

from utils.create_data import Fusion169Dataset, LoadFusion169Dataset
print('Dataset imports OK')
print('All checks passed')
