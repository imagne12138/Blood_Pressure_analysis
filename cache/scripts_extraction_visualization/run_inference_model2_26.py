import os, warnings, numpy as np, torch, torch.nn as nn, h5py
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

class Fusion26Dataset(Dataset):
    def __init__(self, ppg_path, feat_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.ppg_path=ppg_path; self.feat_path=feat_path
        self.ppg_file=None; self.feat_file=None
        self.sbp_mean=sbp_mean; self.sbp_std=sbp_std
        self.dbp_mean=dbp_mean; self.dbp_std=dbp_std
        self.feat_mean=feat_mean; self.feat_std=feat_std
        fi=np.load(indices_dir)
        self.indices=fi["train_idx"] if train else fi["val_idx"]
    def _init_file(self):
        if self.ppg_file is None:
            self.ppg_file=h5py.File(self.ppg_path,"r")
            self.ppg=self.ppg_file["ppg"]; self.sbp=self.ppg_file["sbp"]; self.dbp=self.ppg_file["dbp"]
        if self.feat_file is None:
            self.feat_file=h5py.File(self.feat_path,"r"); self.feats=self.feat_file["ppg_features"]
    def __getitem__(self, index):
        self._init_file(); idx=self.indices[index]
        ppg=self.ppg[idx]; xp=torch.from_numpy(ppg).unsqueeze(0).float()
        fa=self.feats[idx].astype(np.float32)
        if self.feat_mean is not None: fa=(fa-self.feat_mean)/(self.feat_std+1e-8)
        sb=self.sbp[idx]; db=self.dbp[idx]
        if self.sbp_mean is not None:
            sb=(sb-self.sbp_mean)/self.sbp_std; db=(db-self.dbp_mean)/self.dbp_std
        return (xp,torch.from_numpy(fa).float()),torch.tensor([sb,db],dtype=torch.float32)
    def __len__(self): return len(self.indices)

# Correct model matching the checkpoint: cat(context, proj_feature) only = 256+16=272
class FusionModel26Infer(nn.Module):
    def __init__(self, filters=(1,32,64,128), num_layers=2, hidden_dim=128, drop_prob=0.2):
        super().__init__()
        self.dropout=nn.Dropout(drop_prob); self.softmax=nn.Softmax(dim=1)
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(filters[i],filters[i+1],3,1,1),nn.BatchNorm1d(filters[i+1]),nn.ReLU(),nn.Dropout(drop_prob),nn.MaxPool1d(2,2)) for i in range(len(filters)-1)])
        self.bilstm=nn.LSTM(128,hidden_dim,num_layers,batch_first=True,dropout=drop_prob,bidirectional=True)
        self.sbp_attn=nn.Sequential(nn.Linear(hidden_dim*2,1),nn.Tanh())
        self.dbp_attn=nn.Sequential(nn.Linear(hidden_dim*2,1),nn.Tanh())
        self.feature_projection=nn.Sequential(nn.Linear(26,16),nn.BatchNorm1d(16),nn.ReLU(),nn.Dropout(0.1))
        fusion_dim=hidden_dim*2+16  # 256+16=272
        self.linear_sbp_out=nn.Sequential(nn.Linear(fusion_dim,hidden_dim),nn.ReLU(),nn.Dropout(drop_prob),nn.Linear(hidden_dim,1))
        self.linear_dbp_out=nn.Sequential(nn.Linear(fusion_dim,hidden_dim),nn.ReLU(),nn.Dropout(drop_prob),nn.Linear(hidden_dim,1))
    def forward(self,x,features):
        for conv in self.convs: x=conv(x)
        lstm_out,_=self.bilstm(x.permute(0,2,1)); lstm_out=self.dropout(lstm_out)
        e_sbp=self.softmax(self.sbp_attn(lstm_out)); e_dbp=self.softmax(self.dbp_attn(lstm_out))
        c_sbp=torch.sum(e_sbp*lstm_out,dim=1); c_dbp=torch.sum(e_dbp*lstm_out,dim=1)
        pf=self.feature_projection(features)
        c_sbp=torch.cat([c_sbp,pf],dim=1); c_dbp=torch.cat([c_dbp,pf],dim=1)
        return torch.cat([self.linear_sbp_out(c_sbp),self.linear_dbp_out(c_dbp)],dim=1)

CACHE=r"E:\Kaggle_projects\Blood_Pressure_analysis\cache"
BASE=r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset"
CKPT_DIR=os.path.join(CACHE,"model2_fusion_26dim","fusion_feature_proj")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {DEVICE}")

all_p=[]; all_l=[]
for fold in range(5):
    ckpt=os.path.join(CKPT_DIR,f"fusion26_fold_{fold}_feat_proj.pkl")
    ppg_path=os.path.join(BASE,"segmented_records.h5"); feat_path=os.path.join(BASE,"ppg_features.h5")
    fold_dir=os.path.join(BASE,f"cv_fold_{fold}.npz")
    train_ds=Fusion26Dataset(ppg_path,feat_path,fold_dir,train=True)
    with h5py.File(ppg_path,"r") as f:
        ti=train_ds.indices; s=f["sbp"][ti]; d=f["dbp"][ti]
        sbp_m=float(s.mean()); sbp_s=float(s.std()); dbp_m=float(d.mean()); dbp_s=float(d.std())
    with h5py.File(feat_path,"r") as f:
        fe=f["ppg_features"][train_ds.indices]; feat_m=fe.mean(axis=0).astype(np.float32); feat_s=fe.std(axis=0).astype(np.float32)
    ds=Fusion26Dataset(ppg_path,feat_path,fold_dir,train=False,
                       sbp_mean=sbp_m,sbp_std=sbp_s,dbp_mean=dbp_m,dbp_std=dbp_s,
                       feat_mean=feat_m,feat_std=feat_s)
    loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=0)
    model=FusionModel26Infer().to(DEVICE)
    state=torch.load(ckpt,map_location=DEVICE,weights_only=True)
    model.load_state_dict(state); model.eval()
    fp=[]; fl=[]
    with torch.no_grad():
        for (ppg,feats),labels in loader:
            ppg,feats,labels=ppg.to(DEVICE),feats.to(DEVICE),labels.to(DEVICE)
            out=model(ppg,feats)
            out[:,0]=out[:,0]*sbp_s+sbp_m; out[:,1]=out[:,1]*dbp_s+dbp_m
            ld=labels.clone(); ld[:,0]=labels[:,0]*sbp_s+sbp_m; ld[:,1]=labels[:,1]*dbp_s+dbp_m
            fp.append(out.cpu().numpy()); fl.append(ld.cpu().numpy())
    all_p.append(np.concatenate(fp)); all_l.append(np.concatenate(fl))
    print(f"Fold {fold}: {len(all_p[-1])} samples")

pred=np.concatenate(all_p,axis=0); label=np.concatenate(all_l,axis=0)
print(f"Total: {len(pred)} samples")
np.savez(os.path.join(CACHE,"inference_results_model2_26.npz"),pred=pred,label=label)
print("Saved!")
