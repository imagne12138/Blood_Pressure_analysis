import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CACHE=r"E:\Kaggle_projects\Blood_Pressure_analysis\cache"
FIG_DIR=os.path.join(CACHE,"figures"); os.makedirs(FIG_DIR,exist_ok=True)

# Load inference results
data=np.load(os.path.join(CACHE,"inference_results_model2_26.npz"))
pred=data["pred"]; label=data["label"]
sbp_true=label[:,0]; dbp_true=label[:,1]
sbp_pred=pred[:,0]; dbp_pred=pred[:,1]

# Subsample: 3000 random points per plot for legibility
rng=np.random.RandomState(42)
idx=rng.choice(len(sbp_true),min(3000,len(sbp_true)),replace=False)
st,sp=sbp_true[idx],sbp_pred[idx]
dt,dp=dbp_true[idx],dbp_pred[idx]

# ===== 5. Prediction Scatter Plot =====
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,tt,tp,yl in [(axes[0],st,sp,"SBP"),(axes[1],dt,dp,"DBP")]:
    ax.scatter(tt,tp,s=8,alpha=0.3,edgecolors="none",c="#2196F3" if yl=="SBP" else "#FF5722")
    lims=[min(tt.min(),tp.min())-5,max(tt.max(),tp.max())+5]
    ax.plot(lims,lims,"k--",lw=1.5,alpha=0.7)
    # Fit regression line
    slope,intercept,r_val,_,_=stats.linregress(tt,tp)
    x_fit=np.linspace(*lims,100)
    ax.plot(x_fit,slope*x_fit+intercept,"r-",lw=1.5,alpha=0.8)
    mae=np.mean(np.abs(tt-tp)); corr=np.corrcoef(tt,tp)[0,1]
    ax.text(0.05,0.95,f"MAE={mae:.2f}\nR={corr:.3f}",transform=ax.transAxes,fontsize=11,
            verticalalignment="top",bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.8))
    ax.set_xlabel(f"True {yl} (mmHg)",fontsize=12); ax.set_ylabel(f"Predicted {yl} (mmHg)",fontsize=12)
    ax.set_title(f"{yl}: Predicted vs True",fontsize=13,fontweight="bold")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
plt.suptitle("Model2 + 26-dim Fusion: Prediction Scatter Plots",fontsize=14,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"07_scatter_pred_vs_true.png"),dpi=200,bbox_inches="tight")
plt.close(); print("Saved 07_scatter_pred_vs_true.png")

# ===== 6. Bland-Altman Plot =====
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,tt,tp,yl in [(axes[0],st,sp,"SBP"),(axes[1],dt,dp,"DBP")]:
    mean_val=(tt+tp)/2; diff=tt-tp
    md=np.mean(diff); sd=np.std(diff)
    loa_hi=md+1.96*sd; loa_lo=md-1.96*sd
    ax.scatter(mean_val,diff,s=8,alpha=0.3,edgecolors="none",c="#2196F3" if yl=="SBP" else "#FF5722")
    ax.axhline(md,color="k",lw=1.5,linestyle="--",label=f"Mean diff={md:.2f}")
    ax.axhline(loa_hi,color="r",lw=1,linestyle=":",label=f"+1.96SD={loa_hi:.2f}")
    ax.axhline(loa_lo,color="r",lw=1,linestyle=":",label=f"-1.96SD={loa_lo:.2f}")
    ax.legend(fontsize=9)
    ax.set_xlabel(f"Mean of True and Predicted {yl} (mmHg)",fontsize=12)
    ax.set_ylabel(f"True - Predicted {yl} (mmHg)",fontsize=12)
    ax.set_title(f"{yl}: Bland-Altman Plot",fontsize=13,fontweight="bold")
    ax.grid(alpha=0.3)
plt.suptitle("Model2 + 26-dim Fusion: Bland-Altman Plots",fontsize=14,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"08_bland_altman.png"),dpi=200,bbox_inches="tight")
plt.close(); print("Saved 08_bland_altman.png")

# ===== 7. Residual Histogram =====
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,tt,tp,yl in [(axes[0],sbp_true,sbp_pred,"SBP"),(axes[1],dbp_true,dbp_pred,"DBP")]:
    res=tt-tp
    ax.hist(res,bins=120,density=True,alpha=0.7,color="#2196F3" if yl=="SBP" else "#FF5722",edgecolor="white",linewidth=0.3)
    mu,sigma=np.mean(res),np.std(res)
    xs=np.linspace(res.min(),res.max(),200)
    ax.plot(xs,stats.norm.pdf(xs,mu,sigma),"k-",lw=2,label=f"N({mu:.1f},{sigma:.1f})")
    ax.axvline(0,color="gray",ls="--",lw=1,alpha=0.6)
    ax.set_xlabel("Prediction Error (mmHg)",fontsize=12); ax.set_ylabel("Density",fontsize=12)
    ax.set_title(f"{yl}: Error Distribution (µ={mu:.2f}, σ={sigma:.2f})",fontsize=12,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.suptitle("Model2 + 26-dim Fusion: Residual Distributions",fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"09_residual_histogram.png"),dpi=200,bbox_inches="tight")
plt.close(); print("Saved 09_residual_histogram.png")

# ===== 8. Error vs Reference (full data, binned) =====
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,tt,tp,yl in [(axes[0],sbp_true,sbp_pred,"SBP"),(axes[1],dbp_true,dbp_pred,"DBP")]:
    err=np.abs(tt-tp)
    ax.scatter(tt,err,s=1,alpha=0.1,edgecolors="none",c="#2196F3" if yl=="SBP" else "#FF5722")
    # Binned mean: divide into 20 bins
    bins=np.linspace(tt.min(),tt.max(),25)
    bin_centers=(bins[:-1]+bins[1:])/2
    bin_means=[np.mean(err[(tt>=bins[i])&(tt<bins[i+1])]) for i in range(len(bins)-1)]
    ax.plot(bin_centers,bin_means,"r-o",lw=2,markersize=4,label="Binned mean")
    # Linear fit
    slope,intercept,r_val,_,_=stats.linregress(tt,err)
    ax.plot(bins,slope*bins+intercept,"k--",lw=1.5,alpha=0.6,label=f"R={r_val:.3f}")
    ax.set_xlabel(f"True {yl} (mmHg)",fontsize=12); ax.set_ylabel("|Prediction Error| (mmHg)",fontsize=12)
    ax.set_title(f"{yl}: Absolute Error vs True BP",fontsize=12,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.suptitle("Model2 + 26-dim Fusion: Error vs Reference",fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"10_error_vs_reference.png"),dpi=200,bbox_inches="tight")
plt.close(); print("Saved 10_error_vs_reference.png")

print("All plots generated!")
