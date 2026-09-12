"""Verify both new models."""
import sys; sys.path.insert(0, 'e:/Kaggle_projects/Blood_Pressure_analysis')
import torch

from model.Fusion_Opt import FusionModelOpt
m1 = FusionModelOpt(filters=[1,32,64,128], num_layers=2)
print(f'FusionModelOpt: {sum(p.numel() for p in m1.parameters()):,} params')
o1 = m1(torch.randn(4,1,1024), torch.randn(4,17), torch.randn(4,12))
print(f'  Output: {o1.shape}')

from model.Fusion_HeadConcat import FusionHeadConcat26
m2 = FusionHeadConcat26(filters=[1,32,64,128], num_layers=2)
print(f'FusionHeadConcat26: {sum(p.numel() for p in m2.parameters()):,} params')
o2 = m2(torch.randn(4,1,1024), torch.randn(4,26))
print(f'  Output: {o2.shape}')

from utils.create_data import FusionOptDataset, LoadFusionOptDataset
print('Datasets OK')
print('All checks passed')
