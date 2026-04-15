
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

raw = """Row Labels,Mica,Convergent,Discoidal Core,Thinning Flakes,Retouched,Redirecting,Burin/Burin Spall,Burren Adze,Cores,Bipolar,Points,Total Axes,Total Grindstones,Bone Point,Pigmented Exfoliation,Heated,Total Ground HM,Quartzite,Stripey Quartzite,Dark Grey Quartzite,Brown Quartzite,Silcrete,Fine Quartzite,Volcanic,Chert,Buff FGS,Gerowie Tuff,Glass,Qtz Defs 3mm,Translucent qtz defs 3mm,Crystal qtz 3mm,Exotics
0.1,0.03,0.00,0.00,0.28,0.03,0.00,0.00,0.00,0.00,1.34,0.19,0.09,0.22,1.68,1.46,0.09,2.64,5.93,0.00,0.06,0.06,0.06,7.76,1.21,0.25,0.00,0.09,0.87,19.13,54.32,13.54,22.33
0.2,0.00,0.00,0.00,0.92,0.00,0.00,0.00,0.05,0.05,1.42,0.60,0.37,1.01,1.06,3.26,0.14,3.49,11.31,0.00,0.00,0.09,0.05,7.90,1.47,1.06,0.14,0.28,0.00,16.87,59.28,19.99,10.11
0.3,0.00,0.00,0.00,1.52,0.14,0.00,0.00,0.07,0.00,0.55,0.90,0.28,1.18,0.69,9.76,0.07,2.98,11.08,0.00,0.07,0.00,0.14,7.27,2.63,2.15,0.00,0.42,0.07,15.37,46.40,23.82,11.43
0.4,0.07,0.00,0.00,0.41,0.00,0.00,0.00,0.00,0.07,0.34,0.48,0.07,0.61,1.16,7.90,0.14,3.88,8.24,0.00,0.07,0.14,0.00,5.38,3.47,0.75,0.00,0.00,0.07,7.08,34.24,12.80,8.10
0.5,0.04,0.00,0.00,0.08,0.00,0.04,0.00,0.00,0.00,1.04,0.00,0.04,0.00,1.00,2.70,0.04,2.24,2.90,0.00,0.00,0.04,0.08,1.54,1.95,0.33,0.00,0.04,0.00,14.48,44.69,27.18,2.99
0.6,0.02,0.00,0.00,0.00,0.00,0.06,0.00,0.00,0.00,0.61,0.05,0.03,0.18,0.14,1.06,0.03,1.50,1.22,0.00,0.00,0.00,0.02,0.26,0.59,0.21,0.00,0.00,0.00,13.94,51.54,19.79,0.69
0.7,0.00,0.02,0.00,0.00,0.02,0.00,0.00,0.00,0.00,1.01,0.02,0.00,0.08,0.04,0.04,0.06,1.41,0.32,0.00,0.02,0.00,0.04,0.02,0.40,0.56,0.00,0.00,0.00,23.04,77.76,36.26,0.73
0.8,0.00,0.00,0.00,0.00,0.04,0.04,0.00,0.00,0.02,0.33,0.00,0.02,0.11,0.02,0.15,0.09,2.24,0.67,0.00,0.00,0.02,0.07,0.22,1.26,1.04,0.02,0.02,0.00,13.37,39.05,17.74,2.05
0.9,0.00,0.00,0.00,0.00,0.06,0.02,0.00,0.00,0.00,0.54,0.00,0.00,0.09,0.00,0.26,0.09,2.38,0.69,0.00,0.00,0.00,0.17,0.17,0.45,1.88,0.00,0.00,0.00,23.90,86.60,33.32,2.51
1.0,0.04,0.00,0.00,0.00,0.11,0.00,0.00,0.00,0.00,0.64,0.00,0.32,0.11,0.00,0.39,0.11,2.44,1.87,0.00,0.00,0.04,0.18,0.25,0.88,1.02,0.00,0.00,0.00,15.19,32.61,7.67,2.05
1.1,0.00,0.00,0.00,0.00,0.03,0.03,0.00,0.00,0.05,0.62,0.00,0.00,0.41,0.00,0.51,0.08,6.37,2.03,0.00,0.00,0.03,0.54,0.51,0.65,1.00,0.00,0.00,0.00,28.62,53.36,13.28,2.28
1.2,0.05,0.00,0.00,0.00,0.10,0.03,0.00,0.00,0.05,0.61,0.00,0.00,0.43,0.00,0.33,0.08,6.47,4.10,0.00,0.05,0.15,0.97,0.82,2.09,1.15,0.00,0.00,0.00,18.65,36.79,7.87,4.56
1.3,0.00,0.00,0.00,0.00,0.05,0.21,0.05,0.00,0.09,1.27,0.00,0.00,0.28,0.00,0.19,0.09,4.65,2.78,0.00,0.00,0.11,0.97,0.93,1.33,1.76,0.00,0.00,0.00,26.76,55.10,18.30,4.52
1.4,0.04,0.00,0.00,0.00,0.14,0.06,0.00,0.00,0.06,0.88,0.00,0.04,0.33,0.00,0.16,0.16,3.44,3.62,0.00,0.12,0.23,1.41,1.93,1.86,2.38,0.00,0.00,0.00,27.22,40.59,10.49,7.13
1.5,0.00,0.00,0.00,0.00,0.10,0.06,0.00,0.00,0.10,0.90,0.00,0.04,0.06,0.00,0.23,0.17,2.14,3.32,0.00,0.11,0.19,1.39,1.22,1.36,3.04,0.00,0.00,0.00,25.41,51.30,8.84,6.41
1.6,0.07,0.00,0.00,0.00,0.04,0.02,0.02,0.00,0.11,1.35,0.00,0.04,0.22,0.00,0.15,0.15,2.18,3.20,0.00,0.07,0.11,0.65,0.63,1.20,1.72,0.00,0.00,0.00,25.29,49.06,8.42,3.74
1.7,0.05,0.00,0.00,0.00,0.07,0.04,0.04,0.00,0.04,0.90,0.00,0.00,0.27,0.00,0.53,0.16,3.21,2.11,0.02,0.05,0.07,0.35,0.05,0.58,1.49,0.14,0.00,0.00,33.63,61.09,11.96,2.41
1.8,0.05,0.00,0.00,0.00,0.00,0.03,0.00,0.00,0.05,0.51,0.00,0.05,0.15,0.00,1.96,0.15,3.94,2.29,0.00,0.03,0.05,0.13,0.03,0.62,1.52,0.00,0.00,0.00,28.94,56.47,8.75,2.19
1.9,0.07,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.00,0.15,0.00,0.51,0.11,3.74,4.04,0.00,0.00,0.22,0.15,0.26,0.59,0.92,0.04,0.00,0.00,21.15,40.82,5.25,1.87
2.0,0.11,0.00,0.00,0.00,0.05,0.03,0.00,0.00,0.03,0.27,0.00,0.00,0.32,0.00,0.56,0.16,6.58,3.22,0.00,0.03,0.11,0.29,0.27,0.35,1.01,0.03,0.00,0.00,23.99,56.61,9.49,1.92
2.1,0.08,0.00,0.03,0.06,0.03,0.00,0.00,0.00,0.03,0.14,0.00,0.03,0.28,0.00,0.53,0.14,5.31,3.04,0.00,0.11,0.08,0.53,0.22,1.32,2.50,0.06,0.00,0.00,35.59,61.93,9.64,4.27
2.2,0.22,0.00,0.00,0.56,0.13,0.09,0.00,0.00,0.03,0.38,0.00,0.13,0.47,0.00,0.47,0.22,5.71,9.79,0.00,0.25,0.44,1.88,1.76,3.45,5.43,0.22,0.00,0.00,19.52,48.76,8.00,12.39
2.3,0.19,0.12,0.00,0.64,0.05,0.07,0.00,0.00,0.05,0.10,0.00,0.24,0.64,0.00,0.72,0.19,5.39,11.44,0.02,0.36,0.91,3.55,2.67,7.46,4.12,1.50,0.00,0.00,13.40,29.95,3.62,19.10
2.4,0.24,0.19,0.01,1.78,0.22,0.01,0.00,0.00,0.04,0.54,0.03,0.12,1.09,0.00,0.78,0.12,3.00,16.79,0.06,0.35,1.13,7.54,3.40,4.58,2.64,1.12,0.00,0.00,15.27,34.85,8.53,19.48
2.5,0.26,0.20,0.02,1.14,0.12,0.02,0.00,0.00,0.04,0.32,0.00,0.28,0.48,0.00,1.50,0.16,3.15,22.12,0.12,0.54,2.22,8.66,3.05,1.20,1.80,0.90,0.00,0.00,8.17,26.19,8.03,18.15
2.6,0.38,0.13,0.04,0.97,0.38,0.00,0.00,0.00,0.04,0.21,0.04,0.04,0.55,0.00,1.56,0.08,4.84,26.91,0.00,0.80,3.03,10.65,4.38,0.46,1.39,0.80,0.00,0.00,15.45,45.26,15.87,21.26"""
df = pd.read_csv(StringIO(raw))
df.rename(columns={"Row Labels": "Depth"}, inplace=True)

phase2_min, phase2_max = 2.1, 2.6
phase2_mask = (df["Depth"] >= phase2_min) & (df["Depth"] <= phase2_max)
background_mask = df["Depth"] < phase2_min

pulse_vars = [
    "Quartzite", "Silcrete", "Brown Quartzite", "Chert", "Exotics",
    "Thinning Flakes", "Retouched", "Convergent", "Discoidal Core"
]
nonpulse_vars = [
    "Bipolar", "Cores", "Total Axes", "Heated", "Fine Quartzite",
    "Qtz Defs 3mm", "Translucent qtz defs 3mm", "Crystal qtz 3mm"
]

def line_panels(vars_list, title, outpath):
    n = len(vars_list)
    fig, axes = plt.subplots(math.ceil(n/3), 3, figsize=(14, 3.6 * math.ceil(n/3)), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, var in zip(axes, vars_list):
        ax.plot(df["Depth"], df[var], marker="o", linewidth=1.6)
        ax.axvspan(phase2_min, phase2_max, alpha=0.15)
        ax.set_title(var, fontsize=10)
        ax.set_ylabel("Value")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    axes[-1].set_xlabel("Depth (m)")
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)

line_panels(pulse_vars, "Madjedbebe: Variables with a distinct Phase 2 pulse (Phase 2 shaded)", "madjedbebe_phase2_pulse_panel.png")
line_panels(nonpulse_vars, "Madjedbebe: Variables with no distinct Phase 2 pulse (Phase 2 shaded)", "madjedbebe_phase2_no_pulse_panel.png")

heat_vars = pulse_vars + nonpulse_vars
zmat = []
for var in heat_vars:
    bg = df.loc[background_mask, var]
    mu = bg.mean()
    sd = bg.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        sd = 1.0
    z = (df[var] - mu) / sd
    zmat.append(z.values)
zmat = np.array(zmat)

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(zmat, aspect="auto", interpolation="nearest")
ax.set_yticks(np.arange(len(heat_vars)))
ax.set_yticklabels(heat_vars)
ax.set_xticks(np.arange(len(df)))
ax.set_xticklabels(df["Depth"].astype(str), rotation=90)
phase2_cols = np.where(phase2_mask)[0]
ax.axvline(phase2_cols[0]-0.5, linewidth=1.5)
ax.axvline(phase2_cols[-1]+0.5, linewidth=1.5)
ax.set_title("Madjedbebe: Background-standardised z-scores by depth")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Variable")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("z-score vs 0.1–2.0 m background")
fig.tight_layout()
fig.savefig("madjedbebe_phase2_zscore_heatmap.png", dpi=600, bbox_inches="tight")
plt.close(fig)

def normal_loglik_segment(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return -np.inf
    s2 = np.var(x, ddof=1) if n > 1 else 1.0
    s2 = max(s2, 1e-6)
    return -0.5 * n * (np.log(2*np.pi*s2) + 1)

def changepoint_posterior(y, min_seg=3):
    n = len(y)
    cps = np.arange(min_seg, n - min_seg + 1)
    logps = []
    for cp in cps:
        lp = normal_loglik_segment(y[:cp]) + normal_loglik_segment(y[cp:])
        logps.append(lp)
    logps = np.array(logps)
    logps = logps - logps.max()
    probs = np.exp(logps)
    probs = probs / probs.sum()
    return cps, probs

cp_posteriors = {}
for var in pulse_vars:
    y = StandardScaler().fit_transform(df[[var]]).flatten()
    cps, probs = changepoint_posterior(y, min_seg=3)
    cp_posteriors[var] = probs

mean_probs = np.mean(np.vstack(list(cp_posteriors.values())), axis=0)
cp_depths = []
for cp in cps:
    if cp < len(df):
        d_left = df["Depth"].iloc[cp-1]
        d_right = df["Depth"].iloc[cp]
        cp_depths.append((d_left + d_right)/2)
    else:
        cp_depths.append(df["Depth"].iloc[-1])

fig, ax = plt.subplots(figsize=(10, 5))
for var, probs in cp_posteriors.items():
    ax.plot(cp_depths, probs, linewidth=1, alpha=0.55)
ax.plot(cp_depths, mean_probs, linewidth=2.8, label="Mean posterior across pulse variables")
ax.axvspan(phase2_min, phase2_max, alpha=0.15)
ax.invert_xaxis()
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Posterior probability")
ax.set_title("Madjedbebe: Bayesian single change-point posterior by depth")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("madjedbebe_phase2_bayesian_changepoint.png", dpi=600, bbox_inches="tight")
plt.close(fig)

X = df[heat_vars].values
Xz = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(Xz)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(scores[~phase2_mask, 0], scores[~phase2_mask, 1], label="0.1–2.0 m")
ax.scatter(scores[phase2_mask, 0], scores[phase2_mask, 1], label="Phase 2 (2.1–2.6 m)")
for i, depth in enumerate(df["Depth"]):
    ax.text(scores[i, 0], scores[i, 1], f"{depth:.1f}", fontsize=7)
for i, var in enumerate(heat_vars):
    ax.arrow(0, 0, loadings[i, 0]*2.2, loadings[i, 1]*2.2,
             head_width=0.05, length_includes_head=True, alpha=0.65)
    ax.text(loadings[i, 0]*2.35, loadings[i, 1]*2.35, var, fontsize=8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Madjedbebe: PCA of pulse and non-pulse variables")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.legend(frameon=False)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("madjedbebe_phase2_pca.png", dpi=600, bbox_inches="tight")
plt.close(fig)
