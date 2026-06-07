"""
HR Analytics: Employee Attrition EDA + ML Pipeline
====================================================
Dataset: IBM HR Analytics Employee Attrition & Performance
Target: Attrition (Yes/No)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports & Setup
# ─────────────────────────────────────────────────────────────────────────────
import warnings, os, json, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
PALETTE   = {"No": "#4C9BE8", "Yes": "#E84C4C"}
BLUE      = "#4C9BE8"
RED       = "#E84C4C"
BG        = "#F8F9FA"
DARK      = "#2D3748"
sns.set_theme(style="whitegrid", palette="muted")
OUT       = "/mnt/user-data/outputs/attrition_plots"
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ saved {name}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & initial inspection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  1. DATA LOADING & INSPECTION")
print("="*60)

df = pd.read_csv("/mnt/user-data/uploads/Attrition__1_.csv")
print(f"\nShape : {df.shape}   ({df.shape[0]} employees, {df.shape[1]} features)")
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary stats:\n", df.describe().T)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  2. DATA CLEANING")
print("="*60)

print(f"\nMissing values : {df.isnull().sum().sum()}")
print(f"Duplicate rows : {df.duplicated().sum()}")

# Drop constant / useless columns
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df.drop(columns=DROP_COLS, inplace=True)
print(f"\nDropped constant columns: {DROP_COLS}")
print(f"Remaining columns       : {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature mapping (ordinal decode labels)
# ─────────────────────────────────────────────────────────────────────────────
EDU_MAP   = {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}
ENV_MAP   = {1:"Low",2:"Medium",3:"High",4:"Very High"}
JI_MAP    = {1:"Low",2:"Medium",3:"High",4:"Very High"}
JS_MAP    = {1:"Low",2:"Medium",3:"High",4:"Very High"}
PR_MAP    = {1:"Low",2:"Good",3:"Excellent",4:"Outstanding"}
RS_MAP    = {1:"Low",2:"Medium",3:"High",4:"Very High"}
WLB_MAP   = {1:"Bad",2:"Good",3:"Better",4:"Best"}

label_maps = {
    "Education": EDU_MAP, "EnvironmentSatisfaction": ENV_MAP,
    "JobInvolvement": JI_MAP, "JobSatisfaction": JS_MAP,
    "PerformanceRating": PR_MAP, "RelationshipSatisfaction": RS_MAP,
    "WorkLifeBalance": WLB_MAP
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA – Section A: Target distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3. EDA – TARGET DISTRIBUTION")
print("="*60)

vc = df["Attrition"].value_counts()
pct = (vc / len(df) * 100).round(1)
print(f"\n  No  : {vc['No']}  ({pct['No']}%)")
print(f"  Yes : {vc['Yes']}  ({pct['Yes']}%)")
print(f"\n  Class imbalance ratio 1:{vc['No']//vc['Yes']}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
# Bar
axes[0].bar(["No", "Yes"], [vc["No"], vc["Yes"]],
            color=[PALETTE["No"], PALETTE["Yes"]], edgecolor="white", linewidth=1.5, width=0.5)
for i, (v, p) in enumerate(zip([vc["No"], vc["Yes"]], [pct["No"], pct["Yes"]])):
    axes[0].text(i, v + 15, f"{v}\n({p}%)", ha="center", fontsize=12, fontweight="bold", color=DARK)
axes[0].set_title("Attrition Count", fontsize=14, fontweight="bold", color=DARK)
axes[0].set_ylabel("Employees", color=DARK); axes[0].set_facecolor(BG)
# Pie
axes[1].pie([vc["No"], vc["Yes"]], labels=["No Attrition", "Attrition"],
            colors=[PALETTE["No"], PALETTE["Yes"]], autopct="%1.1f%%",
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Attrition Split", fontsize=14, fontweight="bold", color=DARK)
savefig("01_target_distribution")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EDA – Section B: Numerical distributions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  4. EDA – NUMERICAL FEATURE DISTRIBUTIONS")
print("="*60)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\n  Numerical features ({len(num_cols)}): {num_cols}")

ncols = 4
nrows = int(np.ceil(len(num_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*4), facecolor=BG)
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    for val, color in PALETTE.items():
        subset = df[df["Attrition"] == val][col]
        ax.hist(subset, bins=25, alpha=0.65, color=color, label=val, edgecolor="none")
    ax.set_title(col, fontsize=10, fontweight="bold", color=DARK)
    ax.set_facecolor(BG); ax.legend(fontsize=8)
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numerical Feature Distributions by Attrition", fontsize=16, fontweight="bold", y=1.01, color=DARK)
savefig("02_numerical_distributions")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EDA – Section C: Categorical features vs attrition
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  5. EDA – CATEGORICAL FEATURES VS ATTRITION")
print("="*60)

cat_cols = df.select_dtypes(include="object").columns.drop("Attrition").tolist()
print(f"\n  Categorical features ({len(cat_cols)}): {cat_cols}")

fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=BG)
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ax = axes[i]
    ct = df.groupby([col, "Attrition"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct[["No", "Yes"]].plot(kind="bar", ax=ax, color=[PALETTE["No"], PALETTE["Yes"]],
                                edgecolor="white", linewidth=0.8, width=0.7)
    ax.set_title(f"{col} vs Attrition (%)", fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel(""); ax.set_ylabel("% of group"); ax.set_facecolor(BG)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(["No Attrition","Attrition"], fontsize=8)
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Categorical Features vs Attrition Rate", fontsize=16, fontweight="bold", y=1.01, color=DARK)
savefig("03_categorical_vs_attrition")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EDA – Section D: Ordinal satisfaction features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  6. EDA – SATISFACTION & ORDINAL FEATURES")
print("="*60)

ordinal_cols = list(label_maps.keys())
fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=BG)
axes = axes.flatten()
for i, col in enumerate(ordinal_cols):
    ax = axes[i]
    temp = df.copy()
    temp[col+"_label"] = temp[col].map(label_maps[col])
    order = list(label_maps[col].values())
    ct = temp.groupby([col+"_label", "Attrition"]).size().unstack(fill_value=0)
    ct = ct.reindex([o for o in order if o in ct.index])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    if "Yes" in ct_pct.columns:
        ax.bar(ct_pct.index, ct_pct["Yes"], color=RED, edgecolor="white", width=0.5)
    ax.set_title(f"Attrition Rate by {col}", fontsize=10, fontweight="bold", color=DARK)
    ax.set_ylabel("Attrition %"); ax.set_facecolor(BG)
    ax.tick_params(axis="x", rotation=30)
axes[-1].set_visible(False)
fig.suptitle("Attrition Rate across Ordinal/Satisfaction Features", fontsize=16, fontweight="bold", y=1.01, color=DARK)
savefig("04_satisfaction_attrition_rates")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EDA – Section E: Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  7. EDA – CORRELATION HEATMAP")
print("="*60)

df_num = df[num_cols].copy()
df_num["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)
corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(16, 13), facecolor=BG)
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}, annot_kws={"size":7})
ax.set_title("Correlation Heatmap (incl. Attrition_bin)", fontsize=15, fontweight="bold", color=DARK)
savefig("05_correlation_heatmap")

# Top corr with attrition
attrition_corr = corr["Attrition_bin"].drop("Attrition_bin").abs().sort_values(ascending=False)
print("\n  Top 10 features correlated with Attrition:")
print(attrition_corr.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 9. EDA – Section F: Key business insights
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  8. EDA – KEY BUSINESS INSIGHTS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)

# A: Monthly Income boxplot
ax = axes[0, 0]
df.boxplot(column="MonthlyIncome", by="Attrition", ax=ax,
           boxprops=dict(color=BLUE), medianprops=dict(color=RED, lw=2),
           whiskerprops=dict(color=DARK), capprops=dict(color=DARK))
ax.set_title("Monthly Income vs Attrition", fontweight="bold", color=DARK)
ax.set_xlabel("Attrition"); ax.set_ylabel("Monthly Income"); ax.set_facecolor(BG)
plt.sca(ax); plt.title("Monthly Income vs Attrition", fontweight="bold")

# B: Age distribution violin
ax = axes[0, 1]
parts = ax.violinplot([df[df["Attrition"]=="No"]["Age"], df[df["Attrition"]=="Yes"]["Age"]],
                       positions=[0, 1], showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([PALETTE["No"], PALETTE["Yes"]][i])
    pc.set_alpha(0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Attrition", "Attrition"])
ax.set_title("Age Distribution vs Attrition", fontweight="bold", color=DARK)
ax.set_ylabel("Age"); ax.set_facecolor(BG)

# C: OverTime attrition
ax = axes[0, 2]
ot = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
ot_pct = ot.div(ot.sum(axis=1), axis=0) * 100
ot_pct["Yes"].plot(kind="bar", ax=ax, color=RED, edgecolor="white", width=0.4)
ax.set_title("Attrition Rate by OverTime", fontweight="bold", color=DARK)
ax.set_ylabel("Attrition %"); ax.set_xlabel("OverTime"); ax.set_facecolor(BG)
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x()+p.get_width()/2, p.get_height()+0.5),
                ha="center", fontsize=11, fontweight="bold")

# D: Distance from Home
ax = axes[1, 0]
bins = [0, 5, 10, 15, 20, 30]
labels_b = ["0-5", "6-10", "11-15", "16-20", "21+"]
df["DistBin"] = pd.cut(df["DistanceFromHome"], bins=bins, labels=labels_b)
dist_attr = df.groupby(["DistBin", "Attrition"]).size().unstack(fill_value=0)
dist_pct = dist_attr.div(dist_attr.sum(axis=1), axis=0) * 100
if "Yes" in dist_pct.columns:
    dist_pct["Yes"].plot(kind="bar", ax=ax, color=BLUE, edgecolor="white", width=0.5)
ax.set_title("Attrition Rate by Distance from Home", fontweight="bold", color=DARK)
ax.set_ylabel("Attrition %"); ax.set_xlabel("Distance (km)"); ax.set_facecolor(BG)
ax.tick_params(axis="x", rotation=0)

# E: Job Role attrition rate
ax = axes[1, 1]
jr = df.groupby(["JobRole", "Attrition"]).size().unstack(fill_value=0)
jr_pct = jr.div(jr.sum(axis=1), axis=0) * 100
if "Yes" in jr_pct.columns:
    jr_pct["Yes"].sort_values().plot(kind="barh", ax=ax, color=RED, edgecolor="white")
ax.set_title("Attrition Rate by Job Role", fontweight="bold", color=DARK)
ax.set_xlabel("Attrition %"); ax.set_facecolor(BG)

# F: YearsAtCompany vs Attrition (bins)
ax = axes[1, 2]
df["YearsBin"] = pd.cut(df["YearsAtCompany"], bins=[0,1,3,5,10,20,40],
                         labels=["0-1","2-3","4-5","6-10","11-20","20+"])
yc = df.groupby(["YearsBin", "Attrition"]).size().unstack(fill_value=0)
yc_pct = yc.div(yc.sum(axis=1), axis=0) * 100
if "Yes" in yc_pct.columns:
    yc_pct["Yes"].plot(kind="bar", ax=ax, color=BLUE, edgecolor="white", width=0.5)
ax.set_title("Attrition Rate by Years at Company", fontweight="bold", color=DARK)
ax.set_ylabel("Attrition %"); ax.set_xlabel("Years"); ax.set_facecolor(BG)
ax.tick_params(axis="x", rotation=0)

fig.suptitle("Key Business Insights – HR Attrition Analysis", fontsize=16, fontweight="bold", y=1.01, color=DARK)
savefig("06_business_insights")

# Clean temp cols
df.drop(columns=["DistBin","YearsBin"], errors="ignore", inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  9. PREPROCESSING")
print("="*60)

# Encode target
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

# Encode all object columns with LabelEncoder
le = LabelEncoder()
cat_features = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_features:
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  LabelEncoded: {col}")

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train size : {X_train.shape[0]}")
print(f"  Test  size : {X_test.shape[0]}")
print(f"  Positive class in test: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Baseline Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  10. BASELINE MODEL COMPARISON")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    results[name] = {
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "test_acc": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    print(f"\n  {name}")
    print(f"    CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Test F1       : {f1_score(y_test, y_pred):.4f}")
    if y_prob is not None:
        print(f"    ROC-AUC       : {roc_auc_score(y_test, y_prob):.4f}")

# Model comparison chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
metrics = ["cv_f1_mean", "test_f1", "test_roc_auc"]
metric_labels = ["CV F1 (5-fold)", "Test F1", "ROC-AUC"]
colors_ = [BLUE, RED, "#48BB78"]
for k, (metric, mlabel, col) in enumerate(zip(metrics, metric_labels, colors_)):
    ax = axes[k]
    vals = [results[n][metric] for n in results]
    bars = ax.barh(list(results.keys()), vals, color=col, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_title(mlabel, fontsize=13, fontweight="bold", color=DARK)
    ax.set_facecolor(BG)
    for bar, v in zip(bars, vals):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                va="center", fontsize=10, fontweight="bold", color=DARK)
fig.suptitle("Baseline Model Comparison", fontsize=16, fontweight="bold", color=DARK)
savefig("07_model_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=BG)
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Attrition","Attrition"])
    disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
    axes[i].set_title(name, fontsize=12, fontweight="bold", color=DARK)
    axes[i].set_facecolor(BG)
fig.suptitle("Confusion Matrices – All Baseline Models", fontsize=15, fontweight="bold", color=DARK)
savefig("08_confusion_matrices")

# ─────────────────────────────────────────────────────────────────────────────
# 13. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
roc_colors = [BLUE, RED, "#48BB78", "#F6AD55"]
for (name, res), color in zip(results.items(), roc_colors):
    if res["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['test_roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.4, label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=15, fontweight="bold", color=DARK)
ax.legend(loc="lower right", fontsize=10); ax.set_facecolor(BG)
savefig("09_roc_curves")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Select Best Model & Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  11. HYPERPARAMETER TUNING – GRADIENT BOOSTING")
print("="*60)

best_name = max(results, key=lambda n: results[n]["test_roc_auc"])
print(f"\n  Best baseline model: {best_name} (ROC-AUC={results[best_name]['test_roc_auc']:.4f})")
print("  → Tuning Gradient Boosting (best AUC among tree ensembles)\n")

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}
gb_base = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(
    gb_base, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\n  Best params : {grid_search.best_params_}")
print(f"  Best CV AUC : {grid_search.best_score_:.4f}")

y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:,1]

print(f"\n  Tuned Model – Test Metrics:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred_best):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob_best):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best,
      target_names=["No Attrition","Attrition"]))

# ─────────────────────────────────────────────────────────────────────────────
# 15. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  12. FEATURE IMPORTANCE")
print("="*60)

feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features:\n", feat_imp.head(15).to_string())

fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
colors_fi = [RED if i < 5 else BLUE for i in range(15)]
feat_imp.head(15).sort_values().plot(kind="barh", ax=ax, color=colors_fi[::-1], edgecolor="white")
ax.set_title("Top 15 Feature Importances\n(Tuned Gradient Boosting)", fontsize=15, fontweight="bold", color=DARK)
ax.set_xlabel("Importance Score"); ax.set_facecolor(BG)
red_p = mpatches.Patch(color=RED, label="Top 5 drivers")
blue_p = mpatches.Patch(color=BLUE, label="Other top features")
ax.legend(handles=[red_p, blue_p], fontsize=10)
savefig("10_feature_importance")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Tuned model confusion matrix + ROC
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Attrition","Attrition"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Tuned GB – Confusion Matrix", fontsize=13, fontweight="bold", color=DARK)
axes[0].set_facecolor(BG)

fpr, tpr, _ = roc_curve(y_test, y_prob_best)
axes[1].plot(fpr, tpr, color=RED, lw=2.5, label=f"AUC={roc_auc_score(y_test,y_prob_best):.3f}")
axes[1].fill_between(fpr, tpr, alpha=0.1, color=RED)
axes[1].plot([0,1],[0,1],"k--", lw=1, alpha=0.4)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("Tuned GB – ROC Curve", fontsize=13, fontweight="bold", color=DARK)
axes[1].legend(fontsize=11); axes[1].set_facecolor(BG)
fig.suptitle("Best Model Evaluation", fontsize=15, fontweight="bold", color=DARK)
savefig("11_best_model_evaluation")

# ─────────────────────────────────────────────────────────────────────────────
# 17. Save model & scaler
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
joblib.dump(best_model, "/mnt/user-data/outputs/best_model.pkl")
joblib.dump(scaler, "/mnt/user-data/outputs/scaler.pkl")
joblib.dump(X.columns.tolist(), "/mnt/user-data/outputs/feature_names.pkl")
joblib.dump(encoders, "/mnt/user-data/outputs/encoders.pkl")

# Save feature names for UI
feature_names = X.columns.tolist()

# Save a summary JSON for the demo UI
summary = {
    "model": "GradientBoostingClassifier (tuned)",
    "best_params": grid_search.best_params_,
    "cv_roc_auc": round(grid_search.best_score_, 4),
    "test_roc_auc": round(roc_auc_score(y_test, y_prob_best), 4),
    "test_f1": round(f1_score(y_test, y_pred_best), 4),
    "test_accuracy": round(accuracy_score(y_test, y_pred_best), 4),
    "top_features": feat_imp.head(10).to_dict(),
    "feature_names": feature_names
}
with open("/mnt/user-data/outputs/model_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ Model + scaler + summary saved to /mnt/user-data/outputs/")
print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
print(f"""
  Final Model       : Gradient Boosting (tuned via GridSearchCV)
  Best CV ROC-AUC   : {grid_search.best_score_:.4f}
  Test ROC-AUC      : {roc_auc_score(y_test, y_prob_best):.4f}
  Test F1 Score     : {f1_score(y_test, y_pred_best):.4f}
  Test Accuracy     : {accuracy_score(y_test, y_pred_best):.4f}
  Best Params       : {grid_search.best_params_}
  Top 3 Features    : {', '.join(feat_imp.head(3).index.tolist())}
""")
