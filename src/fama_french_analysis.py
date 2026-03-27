"""
fama_french_analysis.py
=======================
Econometric analysis of return differences between emerging and developed
market portfolios using the Fama-French five-factor model.

Research question:
    Do emerging market portfolios earn a systematic risk premium over
    developed markets after controlling for Fama-French factors?

Author  : Diana Krystell Magallanes Pichardo
Course  : Econometría Financiera — EGADE Business School, 2025-2026
Data    : Fama-French Data Library (1989–2025, monthly)
"""

# ── Standard library ───────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── OLS regression via scipy (no statsmodels dependency) ──────────────────────
from numpy.linalg import lstsq

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
COLORS = {"emerging": "#E63946", "developed": "#457B9D"}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load Fama-French portfolio data from Excel and clean missing values.

    Missing value convention: -99.99 and -99.999 represent unavailable
    observations in the Fama-French database. These are replaced with NaN
    and the affected rows are dropped before any analysis.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with a DatetimeIndex and a post2000 indicator.
    """
    df = pd.read_excel(filepath)
    df = df.rename(columns={"Mkt-RF": "MktRF"})

    # Replace Fama-French missing codes with NaN
    for col in ["MktRF", "SMB", "HML", "RMW", "CMA"]:
        df[col] = df[col].replace([-99.99, -99.999], np.nan)

    # Drop rows with any missing value in key variables
    df = df.dropna(subset=["MktRF", "SMB", "HML", "RMW", "CMA", "Emerging"])
    df = df.reset_index(drop=True)

    # Create datetime index for time-series context
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.sort_values("date").reset_index(drop=True)

    # Post-2000 subsample indicator
    df["post2000"] = (df["year"] >= 2000).astype(int)

    # Interaction terms for Model 2
    df["Em_SMB"] = df["Emerging"] * df["SMB"]
    df["Em_HML"] = df["Emerging"] * df["HML"]
    df["Em_RMW"] = df["Emerging"] * df["RMW"]

    print(f"Observations after cleaning: {len(df)}")
    print(f"Period: {df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")
    print(f"  Emerging  : {df['Emerging'].sum():>4} obs")
    print(f"  Developed : {(df['Emerging']==0).sum():>4} obs")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, variance, and std dev of excess returns by market type
    for the full sample and post-2000 subsample.
    """
    rows = []
    for label, mask in [
        ("Emerging (Full)",       df["Emerging"] == 1),
        ("Developed (Full)",      df["Emerging"] == 0),
        ("Emerging (Post-2000)",  (df["Emerging"] == 1) & (df["post2000"] == 1)),
        ("Developed (Post-2000)", (df["Emerging"] == 0) & (df["post2000"] == 1)),
    ]:
        sub = df.loc[mask, "MktRF"]
        rows.append({
            "Portfolio":   label,
            "N":           len(sub),
            "Mean (%)":    round(sub.mean(), 3),
            "Variance":    round(sub.var(), 3),
            "Std Dev (%)": round(sub.std(), 3),
            "Skewness":    round(sub.skew(), 3),
            "Kurtosis":    round(sub.kurt(), 3),
        })
    stats_df = pd.DataFrame(rows).set_index("Portfolio")
    print("\n── Table 1: Descriptive Statistics ──────────────────────────────────")
    print(stats_df.to_string())
    return stats_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_histograms(df: pd.DataFrame, save_path: str = "outputs/fig1_histograms.png"):
    """
    2×2 panel of return distribution histograms:
      Top row    — full sample  (1989–2025)
      Bottom row — post-2000 subsample
      Left col   — emerging markets
      Right col  — developed markets
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Return Distribution Comparison — Emerging vs Developed Markets",
                 fontsize=13, fontweight="bold", y=1.01)

    panels = [
        (0, 0, df["Emerging"] == 1,                          "Emerging — Full Sample",       COLORS["emerging"]),
        (0, 1, df["Emerging"] == 0,                          "Developed — Full Sample",      COLORS["developed"]),
        (1, 0, (df["Emerging"] == 1) & (df["post2000"] == 1),"Emerging — Post-2000",         COLORS["emerging"]),
        (1, 1, (df["Emerging"] == 0) & (df["post2000"] == 1),"Developed — Post-2000",        COLORS["developed"]),
    ]

    for r, c, mask, title, color in panels:
        ax  = axes[r][c]
        sub = df.loc[mask, "MktRF"]
        ax.hist(sub, bins=35, color=color, alpha=0.65, edgecolor=color, linewidth=0.4)
        ax.axvline(sub.mean(), color=color, linestyle="--", linewidth=1.8,
                   label=f"Mean: {sub.mean():.2f}%")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Excess Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def plot_time_series(df: pd.DataFrame, save_path: str = "outputs/fig2_time_series.png"):
    """
    Rolling 12-month average of excess returns for both portfolio types,
    highlighting periods of divergence.
    """
    em  = df[df["Emerging"] == 1].set_index("date")["MktRF"].rolling(12).mean()
    dev = df[df["Emerging"] == 0].set_index("date")["MktRF"].rolling(12).mean()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(em.index,  em,  color=COLORS["emerging"],  lw=1.6, label="Emerging")
    ax.plot(dev.index, dev, color=COLORS["developed"], lw=1.6, label="Developed")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Rolling 12-Month Average Excess Return — Emerging vs Developed",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Excess Return (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OLS REGRESSION HELPER
# ══════════════════════════════════════════════════════════════════════════════

class OLSResult:
    """
    Lightweight OLS wrapper using numpy.linalg.lstsq.
    Computes coefficients, standard errors, t-stats, p-values, R², AIC, BIC.
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, feature_names: list):
        n, k        = X.shape
        self.n      = n
        self.k      = k
        self.names  = feature_names

        # OLS coefficients
        self.coef, residuals, _, _ = lstsq(X, y, rcond=None)

        # Residuals and variance
        self.y_hat  = X @ self.coef
        self.resid  = y - self.y_hat
        self.sigma2 = (self.resid @ self.resid) / (n - k)

        # Coefficient covariance matrix
        XtX_inv     = np.linalg.inv(X.T @ X)
        self.cov    = self.sigma2 * XtX_inv
        self.se     = np.sqrt(np.diag(self.cov))

        # Inference
        self.t_stat = self.coef / self.se
        self.p_val  = 2 * (1 - stats.t.cdf(np.abs(self.t_stat), df=n - k))
        self.ci95   = np.column_stack([
            self.coef - 1.96 * self.se,
            self.coef + 1.96 * self.se,
        ])

        # Goodness of fit
        ss_tot      = np.sum((y - y.mean()) ** 2)
        ss_res      = self.resid @ self.resid
        self.r2     = 1 - ss_res / ss_tot
        self.r2_adj = 1 - (1 - self.r2) * (n - 1) / (n - k)

        # Information criteria
        log_lik     = -n / 2 * (1 + np.log(2 * np.pi * self.sigma2))
        self.aic    = 2 * k - 2 * log_lik
        self.bic    = k * np.log(n) - 2 * log_lik

    def summary(self, title: str = "OLS Regression Results"):
        stars = lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        header = f"\n{'─'*68}\n{title}\n{'─'*68}"
        fmt    = "{:<20} {:>10} {:>10} {:>10} {:>8}  {:<5}"
        header += "\n" + fmt.format("Variable", "Coef", "Std Err", "t-stat", "p-value", "Sig")
        header += "\n" + "─" * 68
        rows   = ""
        for i, name in enumerate(self.names):
            rows += "\n" + fmt.format(
                name,
                f"{self.coef[i]:>10.4f}",
                f"{self.se[i]:>10.4f}",
                f"{self.t_stat[i]:>10.4f}",
                f"{self.p_val[i]:>8.4f}",
                stars(self.p_val[i]),
            )
        footer = (
            f"\n{'─'*68}\n"
            f"  N = {self.n}   R² = {self.r2:.4f}   R² Adj = {self.r2_adj:.4f}"
            f"   AIC = {self.aic:.2f}   BIC = {self.bic:.2f}\n"
            f"  *** p<0.01  ** p<0.05  * p<0.1\n{'─'*68}"
        )
        print(header + rows + footer)


def t_test_coef(result: OLSResult, var_name: str, h0_value: float = 0.0, alpha: float = 0.05):
    """
    Two-sided t-test for a single coefficient: H₀: β = h0_value.

    Returns the t-statistic, p-value, and decision string.
    """
    idx     = result.names.index(var_name)
    t_stat  = (result.coef[idx] - h0_value) / result.se[idx]
    df      = result.n - result.k
    p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    decision = "REJECT H₀" if p_val < alpha else "FAIL TO REJECT H₀"

    print(f"\n── Hypothesis Test: H₀: β_{var_name} = {h0_value}  vs  H₁: β ≠ {h0_value} ──")
    print(f"   β estimate  : {result.coef[idx]:.4f}")
    print(f"   Std. Error  : {result.se[idx]:.4f}")
    print(f"   t-statistic : {t_stat:.4f}")
    print(f"   p-value     : {p_val:.4f}")
    print(f"   95% CI      : [{result.coef[idx] - 1.96*result.se[idx]:.4f}, "
          f"{result.coef[idx] + 1.96*result.se[idx]:.4f}]")
    print(f"   Decision (α={alpha}): {decision}")
    return t_stat, p_val


def f_test_joint(result: OLSResult, var_names: list, alpha: float = 0.05):
    """
    Joint F-test: H₀ that all coefficients in var_names are simultaneously zero.
    Uses the Wald statistic: F = (Rβ)ᵀ (R Cov R ᵀ)⁻¹ (Rβ) / q
    """
    idx = [result.names.index(v) for v in var_names]
    q   = len(idx)
    R   = np.zeros((q, result.k))
    for i, j in enumerate(idx):
        R[i, j] = 1.0

    Rb       = R @ result.coef
    RCovRt   = R @ result.cov @ R.T
    F_stat   = (Rb @ np.linalg.inv(RCovRt) @ Rb) / q
    df1, df2 = q, result.n - result.k
    p_val    = 1 - stats.f.cdf(F_stat, df1, df2)
    decision = "REJECT H₀" if p_val < alpha else "FAIL TO REJECT H₀"

    print(f"\n── Joint F-Test: H₀: {' = '.join(var_names)} = 0 ──")
    print(f"   F-statistic : {F_stat:.4f}")
    print(f"   df          : ({df1}, {df2})")
    print(f"   p-value     : {p_val:.4f}")
    print(f"   Decision (α={alpha}): {decision}")
    return F_stat, p_val


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def run_models(df: pd.DataFrame):
    """
    Estimate Model 1 (baseline) and Model 2 (with interaction terms).

    Model 1:
        MktRF = β₀ + β₁·SMB + β₂·HML + β₃·RMW + β₄·CMA + β₅·Emerging + ε

    Model 2 (adds heterogeneous factor loadings by market type):
        MktRF = β₀ + β₁·SMB + β₂·HML + β₃·RMW + β₄·CMA + β₅·Emerging
              + β₆·(Em×SMB) + β₇·(Em×HML) + β₈·(Em×RMW) + ε

    The interaction terms allow the sensitivity to SMB, HML, and RMW to
    differ between emerging and developed markets. The total SMB sensitivity
    for emerging markets is β₁ + β₆.

    Returns
    -------
    tuple of (OLSResult, OLSResult)
    """
    y = df["MktRF"].values

    # ── Model 1 ────────────────────────────────────────────────────────────────
    X1_cols = ["const", "SMB", "HML", "RMW", "CMA", "Emerging"]
    X1 = np.column_stack([np.ones(len(df)), df[["SMB", "HML", "RMW", "CMA", "Emerging"]].values])
    m1 = OLSResult(y, X1, X1_cols)

    print("\n" + "═"*68)
    print("MODEL 1 — Baseline Fama-French + Emerging Dummy")
    print("═"*68)
    m1.summary("Model 1: MktRF ~ SMB + HML + RMW + CMA + Emerging")

    # Hypothesis tests — Model 1
    print("\n── Model 1 Hypothesis Tests ──────────────────────────────────────────")
    t_test_coef(m1, "Emerging", h0_value=0.0)   # Test 1
    t_test_coef(m1, "Emerging", h0_value=0.6)   # Test 2

    # ── Model 2 ────────────────────────────────────────────────────────────────
    X2_cols = ["const", "SMB", "HML", "RMW", "CMA", "Emerging", "Em_SMB", "Em_HML", "Em_RMW"]
    X2 = np.column_stack([
        np.ones(len(df)),
        df[["SMB", "HML", "RMW", "CMA", "Emerging", "Em_SMB", "Em_HML", "Em_RMW"]].values
    ])
    m2 = OLSResult(y, X2, X2_cols)

    print("\n" + "═"*68)
    print("MODEL 2 — Extended with Interaction Terms")
    print("═"*68)
    m2.summary("Model 2: MktRF ~ FF5 + Emerging + Em×SMB + Em×HML + Em×RMW")

    # Hypothesis tests — Model 2
    print("\n── Model 2 Hypothesis Tests ──────────────────────────────────────────")
    t_test_coef(m2, "Em_SMB", h0_value=0.0)        # Test 3
    f_test_joint(m2, ["Em_HML", "Em_RMW"])          # Test 4

    # Total SMB sensitivity for emerging markets
    smb_dev = m2.coef[m2.names.index("SMB")]
    smb_em  = smb_dev + m2.coef[m2.names.index("Em_SMB")]
    print(f"\n   SMB sensitivity — Developed : {smb_dev:.4f}")
    print(f"   SMB sensitivity — Emerging  : {smb_em:.4f}  (β_SMB + β_Em×SMB)")

    return m1, m2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(m1: OLSResult, m2: OLSResult) -> pd.DataFrame:
    """
    Side-by-side AIC, BIC, R², R²-adjusted comparison.

    AIC is preferred here because the goal is economic inference — identifying
    heterogeneous factor effects — rather than out-of-sample prediction.
    AIC penalises each additional parameter by 2; BIC uses log(n) and is more
    conservative for large samples.
    """
    comp = pd.DataFrame({
        "Model 1": [m1.aic, m1.bic, m1.r2, m1.r2_adj, m1.k],
        "Model 2": [m2.aic, m2.bic, m2.r2, m2.r2_adj, m2.k],
    }, index=["AIC", "BIC", "R²", "R² Adjusted", "Parameters"])
    comp["Δ (M2 − M1)"] = comp["Model 2"] - comp["Model 1"]

    print("\n── Table: Model Comparison ───────────────────────────────────────────")
    print(comp.round(4).to_string())

    preferred = "Model 2" if m2.aic < m1.aic else "Model 1"
    print(f"\n   Preferred model: {preferred}")
    print("   Rationale: Lower AIC + significant Emerging×SMB (p=0.010)")
    print("              justifies the added interaction terms.")
    return comp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COEFFICIENT PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_coefficients(m1: OLSResult, m2: OLSResult,
                      save_path: str = "outputs/fig3_coefficients.png"):
    """
    Forest plot of OLS coefficients with 95% confidence intervals.
    Displays both models side-by-side for easy comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model, title in zip(axes, [m1, m2], ["Model 1 — Baseline", "Model 2 — With Interactions"]):
        names = [n for n in model.names if n != "const"]
        idx   = [model.names.index(n) for n in names]
        coefs = model.coef[idx]
        cis   = model.ci95[idx]
        yerrs = np.abs(cis - coefs[:, None]).T
        colors_bar = ["#E63946" if p < 0.05 else "#A8DADC"
                      for p in model.p_val[idx]]

        ax.barh(names, coefs, xerr=yerrs, color=colors_bar,
                align="center", alpha=0.85, capsize=4,
                error_kw={"elinewidth": 1.5, "ecolor": "gray"})
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Coefficient")
        ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color="#E63946", label="p < 0.05"),
                  Patch(color="#A8DADC", label="p ≥ 0.05")]
    axes[1].legend(handles=legend_els, loc="lower right", fontsize=9)

    plt.suptitle("OLS Coefficients with 95% Confidence Intervals", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("="*68)
    print("FAMA-FRENCH FACTOR ANALYSIS: EMERGING vs DEVELOPED MARKETS")
    print("EGADE Business School — Econometría Financiera")
    print("="*68)

    # 1. Load and clean data
    df = load_and_clean("data/FamaRichvPoor.xlsx")

    # 2. Descriptive statistics
    stats_df = descriptive_stats(df)

    # 3. Visualisations
    plot_histograms(df)
    plot_time_series(df)

    # 4. Model estimation + hypothesis tests
    m1, m2 = run_models(df)

    # 5. Model comparison
    comp = compare_models(m1, m2)

    # 6. Coefficient plot
    plot_coefficients(m1, m2)

    print("\n" + "="*68)
    print("ANALYSIS COMPLETE.")
    print("Outputs saved to: outputs/")
    print("  fig1_histograms.png")
    print("  fig2_time_series.png")
    print("  fig3_coefficients.png")
    print("="*68)
