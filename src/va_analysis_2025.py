"""
DSCI 7000 Capstone: Predicting VA Loan Approval Outcomes
Author: D'Andre Burdin, M.S. Data Science, Hawaii Pacific University

Usage: python src/va_analysis_2025.py [path/to/hmda.csv]

Adapted to the actual 2025 HMDA VA home-purchase extract (99 columns).

Key correction vs. the generic version: nine fields are excluded as target
leakage. Six are structurally missing for all denials (interest_rate,
rate_spread, total_loan_costs, origination_charges, discount_points,
lender_credits) because they describe loan terms that only exist once a loan
is made. Two more are fully populated but encode the outcome directly
(denial_reason, purchaser_type). Including any of them would produce
near-perfect accuracy that means nothing.
"""

import os, sys, warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, f1_score, brier_score_loss, roc_curve,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Download instructions: see data/README.md. Pass a custom path as the first argument.
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "loan_types_3_loan_purposes_1_year_2025.csv")
OUT = "outputs"; FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
RS, TEST_SIZE = 42, 0.25
STATE_SUBSET = None

# Fields excluded because they are determined AFTER or BY the lending decision.
# Verified empirically: each is either structurally missing for denials, or its
# "not applicable" code is assigned only to loans that were never originated.
LEAKAGE = ["interest_rate", "rate_spread", "total_loan_costs", "total_points_and_fees",
           "origination_charges", "discount_points", "lender_credits", "purchaser_type",
           "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
           # hoepa_status: codes 1/2 have 100% approval; code 3 ("not applicable",
           # assigned when no loan is originated) has 23.6%. Univariate AUC 0.987.
           "hoepa_status",
           # initially_payable_to_institution: code 3 shows the same 23.9% pattern.
           "initially_payable_to_institution"]

log = []
def step(name, b, a, note=""):
    log.append({"step": name, "before": b, "after": a, "removed": b - a,
                "pct_removed": round(100*(b-a)/max(b,1), 2), "note": note})
    print(f"  {name}: {b:,} -> {a:,}  (removed {b-a:,})")

MISSING_TOKENS = ["na", "exempt", "", "nan"]
def clean_num(s):
    """Numeric conversion that correctly treats NA/Exempt tokens as missing."""
    out = pd.to_numeric(
        s.astype(str).str.strip().str.lower().replace(MISSING_TOKENS, np.nan),
        errors="coerce")
    return out

# ---------------- 1. LOAD AND FILTER ----------------
print("\n[1] Loading and filtering to study population")
df = pd.read_csv(INPUT_CSV, low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]
n0 = len(df); print(f"  Loaded {n0:,} records, {len(df.columns)} columns")
log.append({"step": "Raw download (VA, home purchase, 2025)", "before": n0,
            "after": n0, "removed": 0, "pct_removed": 0.0, "note": "as downloaded"})

for col, keep, label in [("lien_status", [1], "First lien only"),
                         ("occupancy_type", [1], "Principal residence only"),
                         ("construction_method", [1], "Site-built only")]:
    b = len(df); df = df[df[col].isin(keep)]; step(label, b, len(df), f"{col} in {keep}")

b = len(df); df = df[df["total_units"].astype(str).str.strip() == "1"]
step("Single family only", b, len(df), "total_units == 1")

b = len(df); df = df[df["reverse_mortgage"].isin([2])]
step("Exclude reverse mortgages", b, len(df), "reverse_mortgage == 2 (no)")

b = len(df); df = df[df["business_or_commercial_purpose"].isin([2])]
step("Exclude business purpose", b, len(df), "personal purpose only")

# Preapproval-requested applications must be removed: when such a request is
# denied, HMDA records action_taken = 7, which falls outside our outcome
# definition. Retaining them would leave an approval-only subgroup (11,927
# records, 0 denials) and create perfect separation on the preapproval field.
b = len(df); df = df[df["preapproval"].isin([2])]
step("Exclude preapproval requests", b, len(df),
     "preapproval==2; denied preapprovals are action_taken 7, outside outcome scope")

if STATE_SUBSET:
    b = len(df); df = df[df["state_code"] == STATE_SUBSET]
    step(f"State subset {STATE_SUBSET}", b, len(df), "RQ4")

# ---------------- 2. OUTCOME ----------------
print("\n[2] Defining outcome")
b = len(df); df = df[df["action_taken"].isin([1, 2, 3])]
step("Lender decision only", b, len(df),
     "kept originated(1)/approved-not-accepted(2)/denied(3); dropped withdrawn, incomplete, purchased, preapproval")
df["approved"] = df["action_taken"].isin([1, 2]).astype(int)
print(f"  Approval rate: {df['approved'].mean():.4f} "
      f"({int(df['approved'].sum()):,} approved / {int((1-df['approved']).sum()):,} denied)")

# ---------------- 3. CLEAN NUMERICS ----------------
print("\n[3] Cleaning numeric fields and auditing missingness")
NUMERIC = ["loan_amount", "income", "property_value", "loan_term",
           "loan_to_value_ratio", "tract_minority_population_percent",
           "tract_to_msa_income_percentage", "tract_population",
           "tract_owner_occupied_units", "ffiec_msa_md_median_family_income",
           "tract_median_age_of_housing_units"]
for c in NUMERIC:
    if c in df.columns: df[c] = clean_num(df[c])

miss = df[NUMERIC].isna().mean().sort_values(ascending=False)
print("  Missingness:")
for k, v in miss.items(): print(f"    {k:<40}{v:6.2%}")
miss.rename("pct_missing").to_csv(os.path.join(OUT, "missingness.csv"))
drop_hi = miss[miss > 0.40].index.tolist()
if drop_hi: print(f"  Dropping >40% missing: {drop_hi}")

# ---------------- 4. DTI (RQ2 THRESHOLD) ----------------
print("\n[4] DTI processing and threshold test")
def dti_lb(v):
    v = str(v).strip()
    if v.startswith("<20"): return 10.0
    if v.startswith("20%"): return 20.0
    if v.startswith("30%"): return 30.0
    if v.startswith("50%"): return 50.0
    if v.startswith(">60"): return 61.0
    try: return float(v)
    except ValueError: return np.nan

df["dti_bin"] = df["debt_to_income_ratio"].astype(str).str.strip()
df["dti_numeric"] = df["dti_bin"].apply(dti_lb)
df.loc[df["dti_numeric"].isna(), "dti_bin"] = "Missing"

dti_tbl = (df[df["dti_numeric"].notna()]
           .groupby("dti_bin")
           .agg(n=("approved", "size"),
                denial_rate=("approved", lambda x: 1 - x.mean()),
                lb=("dti_numeric", "first"))
           .sort_values("lb").reset_index())
dti_tbl.to_csv(os.path.join(OUT, "dti_threshold.csv"), index=False)
print("  Denial rate by DTI bin:")
for _, r in dti_tbl.iterrows():
    print(f"    {r['dti_bin']:>10}  n={int(r['n']):>8,}  denial={r['denial_rate']:.4f}")

below = dti_tbl[(dti_tbl.lb >= 45) & (dti_tbl.lb < 50)]
above = dti_tbl[dti_tbl.lb >= 50]
if len(below) and len(above):
    jump = above.denial_rate.iloc[0] - below.denial_rate.iloc[-1]
    print(f"\n  >>> DTI 50% THRESHOLD JUMP: {jump*100:+.1f} percentage points "
          f"({below.dti_bin.iloc[-1]} -> {above.dti_bin.iloc[0]})")
    print(f"      Published literature (St. Louis Fed) reports 15-17 points nationally.")

plt.figure(figsize=(11, 5))
plt.plot(dti_tbl.dti_bin, dti_tbl.denial_rate, marker="o", linewidth=2)
idx50 = np.where(dti_tbl.lb.values >= 50)[0]
if len(idx50): plt.axvline(idx50[0], color="crimson", ls="--", label="50% DTI threshold")
plt.xticks(rotation=45, ha="right"); plt.ylabel("Denial rate")
plt.xlabel("Debt-to-income ratio"); plt.title("VA loan denial rate by DTI bin (2025)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "dti_threshold.png"), dpi=150); plt.close()

# ---------------- 5. OUTLIERS ----------------
print("\n[5] Outlier screening")
b = len(df); df = df[(df.income.isna()) | ((df.income > 0) & (df.income < 10000))]
step("Income plausibility", b, len(df), "0 < income < $10M (reported in $000s)")
b = len(df); df = df[(df.loan_amount > 10000) & (df.loan_amount < 5000000)]
step("Loan amount plausibility", b, len(df), "$10k-$5M")
b = len(df); df = df[(df.loan_to_value_ratio.isna()) |
                     ((df.loan_to_value_ratio > 0) & (df.loan_to_value_ratio <= 150))]
step("LTV plausibility", b, len(df), "0-150%")

# ---------------- 6. FEATURES ----------------
print("\n[6] Feature construction")
print(f"  EXCLUDED as target leakage: {LEAKAGE}")

NUM_FEATS = [c for c in ["loan_amount", "income", "property_value", "loan_term",
                         "loan_to_value_ratio", "dti_numeric",
                         "tract_minority_population_percent",
                         "tract_to_msa_income_percentage",
                         "ffiec_msa_md_median_family_income"]
             if c in df.columns and c not in drop_hi and c not in LEAKAGE]
CAT_FEATS = [c for c in ["dti_bin", "applicant_credit_score_type",
                         "co-applicant_credit_score_type", "aus-1",
                         "conforming_loan_limit", "applicant_age",
                         "submission_of_application", "state_code"]
             if c in df.columns and c not in LEAKAGE]
PROT = [c for c in ["derived_race", "derived_ethnicity", "derived_sex"] if c in df.columns]

for c in CAT_FEATS: df[c] = df[c].astype(str).str.strip()
print(f"  Numeric ({len(NUM_FEATS)}): {NUM_FEATS}")
print(f"  Categorical ({len(CAT_FEATS)}): {CAT_FEATS}")
print(f"  Protected (evaluation only, NOT model inputs): {PROT}")

# Export the screened dataset for triangulation in JMP Pro. This is the exact
# analytic sample used below, so both tools operate on identical data.
jmp_cols = NUM_FEATS + CAT_FEATS + PROT + ["approved", "action_taken"]
df[jmp_cols].to_csv(os.path.join(OUT, "va_screened_for_jmp.csv"), index=False)
print(f"  Exported screened data for JMP: {OUT}/va_screened_for_jmp.csv "
      f"({len(df):,} rows, {len(jmp_cols)} columns)")

# Also export the DTI summary table in a JMP-friendly shape
if os.path.exists(os.path.join(OUT, "dti_threshold.csv")):
    _d = pd.read_csv(os.path.join(OUT, "dti_threshold.csv"))
    _d["denial_rate_pct"] = (_d["denial_rate"] * 100).round(2)
    _d.to_csv(os.path.join(OUT, "dti_for_jmp.csv"), index=False)

X = df[NUM_FEATS + CAT_FEATS]
y = df["approved"]
A = df[PROT]
X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
    X, y, A, test_size=TEST_SIZE, random_state=RS, stratify=y)
print(f"  Train {len(X_tr):,} | Test {len(X_te):,}")

prep = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler())]), NUM_FEATS),
    ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("e", OneHotEncoder(handle_unknown="ignore", min_frequency=100, sparse_output=False))]),
     CAT_FEATS)])

# ---------------- 7. MODELS ----------------
print("\n[7] Fitting models")
models = {
    "Logistic Regression": Pipeline([("p", prep), ("c", LogisticRegression(
        max_iter=3000, random_state=RS, class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", prep), ("c", RandomForestClassifier(
        n_estimators=250, min_samples_leaf=25, n_jobs=-1, random_state=RS,
        class_weight="balanced"))]),
    "Gradient Boosting": Pipeline([("p", prep), ("c", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, random_state=RS))]),
}
res, preds = [], {}
for name, pipe in models.items():
    print(f"  {name}...", flush=True)
    pipe.fit(X_tr, y_tr)
    pr = pipe.predict_proba(X_te)[:, 1]
    pd_ = (pr >= 0.5).astype(int)
    preds[name] = {"proba": pr, "pred": pd_}
    res.append({"model": name, "auc_roc": roc_auc_score(y_te, pr),
                "f1": f1_score(y_te, pd_), "brier": brier_score_loss(y_te, pr),
                "accuracy": (pd_ == y_te).mean()})
    r = res[-1]
    print(f"    AUC={r['auc_roc']:.4f}  F1={r['f1']:.4f}  Brier={r['brier']:.4f}  Acc={r['accuracy']:.4f}")

perf = pd.DataFrame(res); perf.to_csv(os.path.join(OUT, "model_performance.csv"), index=False)
base = perf.loc[perf.model == "Logistic Regression", "auc_roc"].iloc[0]
best = perf.loc[perf.auc_roc.idxmax()]
print(f"\n  >>> RQ2: baseline AUC {base:.4f} | best {best['model']} {best['auc_roc']:.4f} "
      f"(delta {best['auc_roc']-base:+.4f})")

plt.figure(figsize=(6.5, 6.5))
for n in models:
    fpr, tpr, _ = roc_curve(y_te, preds[n]["proba"])
    plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, preds[n]['proba']):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.4); plt.legend(); plt.grid(alpha=.3)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC: VA loan approval prediction"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=150); plt.close()

plt.figure(figsize=(6.5, 6.5))
for n in models:
    fp, mp = calibration_curve(y_te, preds[n]["proba"], n_bins=10)
    plt.plot(mp, fp, marker="o", label=n)
plt.plot([0, 1], [0, 1], "k--", alpha=.4); plt.legend(); plt.grid(alpha=.3)
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed approval frequency")
plt.title("Calibration"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "calibration_curves.png"), dpi=150); plt.close()

# ---------------- 8. INTERPRETABILITY (RQ1) ----------------
print("\n[8] Interpretability")
lr = models["Logistic Regression"]
names = lr.named_steps["p"].get_feature_names_out()
co = lr.named_steps["c"].coef_[0]
odds = pd.DataFrame({"feature": names, "coefficient": co, "odds_ratio": np.exp(co)})
odds = odds.sort_values("coefficient", key=abs, ascending=False)
odds.to_csv(os.path.join(OUT, "logistic_odds_ratios.csv"), index=False)
print("  Top 15 predictors (logistic, by |coef|):")
for _, r in odds.head(15).iterrows():
    print(f"    {r['feature'][:52]:<54} OR={r['odds_ratio']:.3f}")

try:
    import shap
    print("  SHAP on best model (5,000 sampled test rows)...", flush=True)
    bp = models[best["model"]]
    Xs = X_te.sample(min(5000, len(X_te)), random_state=RS)
    Xp = bp.named_steps["p"].transform(Xs)
    if hasattr(Xp, "toarray"): Xp = Xp.toarray()
    ex = shap.Explainer(bp.named_steps["c"], Xp)
    sv = ex(Xp, check_additivity=False)
    imp = pd.DataFrame({"feature": names,
                        "mean_abs_shap": np.abs(sv.values).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(OUT, "shap_importance.csv"), index=False)
    print("  Top 15 by mean |SHAP|:")
    for _, r in imp.head(15).iterrows():
        print(f"    {r['feature'][:52]:<54} {r['mean_abs_shap']:.4f}")
    shap.summary_plot(sv.values, Xp, feature_names=names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_summary.png"), dpi=150,
                                    bbox_inches="tight"); plt.close()
except Exception as e:
    print(f"  SHAP step skipped: {e}")

# ---------------- 9. FAIRNESS STAGE 1 ----------------
print("\n[9] Fairness, stage 1: measurement")
bn = best["model"]; pp = preds[bn]["proba"]
# With a ~92% approval base rate, a 0.50 cutoff approves nearly everyone and
# makes false-positive rates hard to read. Use the Youden-optimal threshold
# (max TPR - FPR) so error rates are informative, and report both.
_fpr, _tpr, _thr = roc_curve(y_te, pp)
THRESH = float(_thr[np.argmax(_tpr - _fpr)])
yp = (pp >= THRESH).astype(int)
print(f"  Decision threshold: {THRESH:.4f} (Youden-optimal; 0.50 approves "
      f"{(pp >= 0.5).mean():.1%} of applicants vs {(pp >= THRESH).mean():.1%} here)")
pd.DataFrame([{"threshold_used": THRESH, "selection_rate_at_0.5": (pp >= 0.5).mean(),
               "selection_rate_at_youden": (pp >= THRESH).mean()}]
             ).to_csv(os.path.join(OUT, "threshold.csv"), index=False)
rows = []
EXCLUDE_GRP = ["race not available", "ethnicity not available", "sex not available",
               "free form text only"]
for attr in PROT:
    g = A_te[attr].fillna("Unknown").astype(str)
    vc = g.value_counts()
    valid = [k for k in vc.index if vc[k] >= 200 and k.strip().lower() not in EXCLUDE_GRP]
    print(f"\n  {attr}:")
    for k in valid:
        m = (g == k).values
        yt, yh, pr = y_te.values[m], yp[m], pp[m]
        tpr = yh[yt == 1].mean() if (yt == 1).sum() else np.nan
        fpr = yh[yt == 0].mean() if (yt == 0).sum() else np.nan
        rows.append({"attribute": attr, "group": k, "n": int(m.sum()),
                     "actual_approval_rate": yt.mean(),
                     "predicted_selection_rate": yh.mean(),
                     "true_positive_rate": tpr, "false_positive_rate": fpr,
                     "mean_predicted_prob": pr.mean(),
                     "calibration_error": abs(pr.mean() - yt.mean())})
        print(f"    {k[:38]:<40} n={int(m.sum()):>7,}  actual={yt.mean():.4f}  "
              f"sel={yh.mean():.4f}  TPR={tpr:.4f}  FPR={fpr:.4f}")

fair = pd.DataFrame(rows); fair.to_csv(os.path.join(OUT, "fairness_metrics.csv"), index=False)
summ = []
print("\n  Fairness summary (max - min across groups):")
for attr in fair.attribute.unique():
    s = fair[fair.attribute == attr]
    dp = s.predicted_selection_rate.max() - s.predicted_selection_rate.min()
    eo = s.true_positive_rate.max() - s.true_positive_rate.min()
    fd = s.false_positive_rate.max() - s.false_positive_rate.min()
    summ.append({"attribute": attr, "demographic_parity_difference": dp,
                 "equal_opportunity_difference": eo, "fpr_difference": fd,
                 "equalized_odds_difference": max(eo, fd),
                 "max_calibration_error": s.calibration_error.max()})
    print(f"    {attr}: DP={dp:.4f}  EqOpp={eo:.4f}  EqOdds={max(eo,fd):.4f}  "
          f"MaxCalErr={s.calibration_error.max():.4f}")
pd.DataFrame(summ).to_csv(os.path.join(OUT, "fairness_summary.csv"), index=False)

# ---------------- 10. FAIRNESS STAGE 2 ----------------
print("\n[10] Fairness, stage 2: decomposition")
print("  Raw gap = observed approval difference vs reference group.")
print("  Explained = difference the model predicts from underwriting factors alone.")
print("  Residual = what underwriting factors do NOT account for.")
dec = []
for attr in PROT:
    g = A_te[attr].fillna("Unknown").astype(str)
    vc = g.value_counts()
    valid = [k for k in vc.index if vc[k] >= 200 and k.strip().lower() not in EXCLUDE_GRP]
    if len(valid) < 2: continue
    ref = valid[0]
    rm = (g == ref).values
    ra, rp = y_te.values[rm].mean(), pp[rm].mean()
    for k in valid[1:]:
        m = (g == k).values
        raw = y_te.values[m].mean() - ra
        exp = pp[m].mean() - rp
        dec.append({"attribute": attr, "group": k, "reference": ref, "n": int(m.sum()),
                    "raw_gap": raw, "explained_by_underwriting": exp,
                    "residual_gap": raw - exp,
                    "pct_explained": 100*exp/raw if raw else np.nan})
d = pd.DataFrame(dec); d.to_csv(os.path.join(OUT, "fairness_decomposition.csv"), index=False)
for _, r in d.iterrows():
    print(f"    {r['attribute'][8:]:<12} {r['group'][:32]:<34} vs {r['reference'][:20]:<22} "
          f"raw={r['raw_gap']:+.4f}  expl={r['explained_by_underwriting']:+.4f}  "
          f"resid={r['residual_gap']:+.4f}")

# ---------------- 11. DESCRIPTIVES ----------------
print("\n[11] Descriptives and screening log")
dr = []
for attr in PROT:
    for k, s in df.groupby(df[attr].astype(str)):
        if len(s) < 200: continue
        dr.append({"attribute": attr, "group": k, "n": len(s),
                   "approval_rate": s.approved.mean(),
                   "denial_rate": 1 - s.approved.mean(),
                   "median_income_000s": s.income.median(),
                   "median_loan_amount": s.loan_amount.median(),
                   "median_ltv": s.loan_to_value_ratio.median()})
pd.DataFrame(dr).to_csv(os.path.join(OUT, "descriptives.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(OUT, "screening_summary.csv"), index=False)

print("\n" + "="*64)
print(f"COMPLETE. Analytic sample: {len(df):,}  |  Approval rate: {df.approved.mean():.4f}")
print("="*64)
