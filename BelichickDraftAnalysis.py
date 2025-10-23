# ============================================================
# BELICHICK DRAFT VALUE ANALYSIS (2000–2023)
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("/Users/anokhpalakurthi/Downloads/2000-2023 NFL Drafts.csv")

# Clean and prepare
df = df[df["Rnd"].ne("Rnd")]
df = df.dropna(subset=["Player", "Tm"])
for col in ["Yr", "Rnd", "Pick", "Age", "DrAV", "wAV"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=["College"], errors="ignore")
df["OutlierFlag"] = ((df["DrAV"] == 0) & (df["wAV"] > 0)).astype(int)
train = df.loc[(df["DrAV"].notna()) & (df["OutlierFlag"] == 0)].copy()

# ============================================================
# 2. EXPECTED VALUE MODEL (LOG-LINEAR)
# ============================================================

X = sm.add_constant(np.log(train["Pick"]))
y = train["DrAV"]
model = sm.OLS(y, X).fit()
print(model.summary())

train["Expected_DrAV"] = model.predict(X)
train["AV_Above_Expected"] = train["DrAV"] - train["Expected_DrAV"]

# ============================================================
# 3. VISUALIZATION — EXPECTED VALUE CURVE
# ============================================================

plt.figure(figsize=(9,6))
sns.scatterplot(x="Pick", y="DrAV", data=train, alpha=0.3, label="Actual")
sns.lineplot(x="Pick", y="Expected_DrAV", data=train, color="red", label="Expected (Model)")
plt.title("Expected Draft AV by Pick Number (Log-Linear Decay)")
plt.xlabel("Draft Pick Number")
plt.ylabel("Approximate Value (DrAV)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 4. TEAM-LEVEL DRAFT PERFORMANCE
# ============================================================

team_perf = (
    train.groupby(["Tm", "Yr"], as_index=False)
    .agg(
        Total_AV=("DrAV", "sum"),
        Expected_AV=("Expected_DrAV", "sum"),
        Picks=("Pick", "count"),
    )
)
team_perf["AV_Above_Expected"] = team_perf["Total_AV"] - team_perf["Expected_AV"]

# ============================================================
# 5. PATRIOTS VS LEAGUE COMPARISON (YEARLY)
# ============================================================

patriots = team_perf[team_perf["Tm"] == "NWE"].copy()
league = (
    team_perf[team_perf["Tm"] != "NWE"]
    .groupby("Yr", as_index=False)
    .agg(League_Mean_AV=("AV_Above_Expected", "mean"),
         League_SD_AV=("AV_Above_Expected", "std"))
)

comparison = pd.merge(patriots, league, on="Yr", how="left")
comparison["Z_Score_vs_League"] = (
    (comparison["AV_Above_Expected"] - comparison["League_Mean_AV"]) /
    comparison["League_SD_AV"]
)

plt.figure(figsize=(10,6))
sns.lineplot(data=comparison, x="Yr", y="AV_Above_Expected", label="Patriots", linewidth=2.5)
sns.lineplot(data=comparison, x="Yr", y="League_Mean_AV", label="League Avg", linestyle="--", color="gray", linewidth=2)
plt.title("Patriots vs. League Draft Value Over Time (2000–2023)")
plt.xlabel("Draft Year")
plt.ylabel("AV Above Expected (Relative to Pick Value)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

t_stat, p_val = ttest_ind(
    patriots["AV_Above_Expected"],
    team_perf[team_perf["Tm"] != "NWE"]["AV_Above_Expected"],
    equal_var=False
)
print(f"\nPatriots vs League (2000–2023): t={t_stat:.3f}, p={p_val:.5f}")

# ============================================================
# 6. DEFINE ERA PHASES
# ============================================================

def assign_phase(year):
    if 2000 <= year <= 2005: return "Phase 1: 2000–2005 (Dynasty Build)"
    elif 2006 <= year <= 2011: return "Phase 2: 2006–2011 (Reload & Adjust)"
    elif 2012 <= year <= 2017: return "Phase 3: 2012–2017 (Sustain & Decline)"
    elif 2018 <= year <= 2023: return "Phase 4: 2018–2023 (Post-Brady / Decline)"
    return None

patriots["Phase"] = patriots["Yr"].apply(assign_phase)
non_pats = team_perf[team_perf["Tm"] != "NWE"].copy()
non_pats["Phase"] = non_pats["Yr"].apply(assign_phase)

# ============================================================
# 7. PHASE-LEVEL ANOVA + VISUALS
# ============================================================

phase_groups = [g["AV_Above_Expected"].dropna() for _, g in patriots.groupby("Phase")]
anova_res = f_oneway(*phase_groups)
print(f"\nBelichick Era ANOVA: F={anova_res.statistic:.3f}, p={anova_res.pvalue:.4f}")

# Pairwise comparisons
phases = patriots["Phase"].unique()
for i in range(len(phases)):
    for j in range(i+1, len(phases)):
        t_res = ttest_ind(
            patriots[patriots["Phase"] == phases[i]]["AV_Above_Expected"],
            patriots[patriots["Phase"] == phases[j]]["AV_Above_Expected"],
            equal_var=False
        )
        print(f"{phases[i]} vs {phases[j]}: t={t_res.statistic:.3f}, p={t_res.pvalue:.4f}")

# Phase aggregation
phase_compare = (
    patriots.groupby("Phase", as_index=False)["AV_Above_Expected"].mean()
    .rename(columns={"AV_Above_Expected": "Patriots_Mean"})
    .merge(
        non_pats.groupby("Phase", as_index=False)["AV_Above_Expected"].mean()
        .rename(columns={"AV_Above_Expected": "League_Mean"}),
        on="Phase", how="outer"
    )
)

phase_order = [
    "Phase 1: 2000–2005 (Dynasty Build)",
    "Phase 2: 2006–2011 (Reload & Adjust)",
    "Phase 3: 2012–2017 (Sustain & Decline)",
    "Phase 4: 2018–2023 (Post-Brady / Decline)"
]
phase_compare["Phase"] = pd.Categorical(phase_compare["Phase"], categories=phase_order, ordered=True)
phase_compare = phase_compare.sort_values("Phase")
phase_compare["Net_Diff"] = phase_compare["Patriots_Mean"] - phase_compare["League_Mean"]

plt.figure(figsize=(10,6))
sns.barplot(data=phase_compare, x="Phase", y="Net_Diff", palette="coolwarm")
plt.axhline(0, color="black", linestyle="--")
plt.title("Net Draft Value Difference (Patriots - League) by Era Phase (2000–2023)")
plt.xlabel("Belichick Era Phase")
plt.ylabel("Net AV Above League Avg")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# 8. Z-SCORE NORMALIZATION (LEAGUE-ADJUSTED)
# ============================================================

team_perf["Z_AV_Above_Expected"] = team_perf.groupby("Yr")["AV_Above_Expected"].transform(lambda x: (x - x.mean()) / x.std())
pats_z = team_perf[team_perf["Tm"] == "NWE"].copy()
nonpats_z = team_perf[team_perf["Tm"] != "NWE"].copy()
pats_z["Phase"] = pats_z["Yr"].apply(assign_phase)
nonpats_z["Phase"] = nonpats_z["Yr"].apply(assign_phase)

phase_z_compare = (
    pats_z.groupby("Phase", as_index=False)["Z_AV_Above_Expected"].mean().rename(columns={"Z_AV_Above_Expected": "Patriots_Z"})
    .merge(
        nonpats_z.groupby("Phase", as_index=False)["Z_AV_Above_Expected"].mean().rename(columns={"Z_AV_Above_Expected": "League_Z"}),
        on="Phase", how="outer"
    )
)

phase_z_compare["Phase"] = pd.Categorical(phase_z_compare["Phase"], categories=phase_order, ordered=True)
phase_z_compare = phase_z_compare.sort_values("Phase")

# Dumbbell chart
plt.figure(figsize=(10,6))
for _, row in phase_z_compare.iterrows():
    plt.plot([row["League_Z"], row["Patriots_Z"]], [row["Phase"], row["Phase"]], color="gray", linewidth=2, alpha=0.7)
    plt.scatter(row["League_Z"], row["Phase"], color="gray", s=80)
    plt.scatter(row["Patriots_Z"], row["Phase"], color="navy", s=80)
plt.axvline(0, color="black", linestyle="--")
plt.title("Patriots vs League Draft Value by Era Phase (Z-Score Normalized)")
plt.xlabel("Normalized Draft Value (Z-Score vs League)")
plt.tight_layout()
plt.show()

# ============================================================
# 9. EXPORT SUMMARY TABLES (Yearly + Phase Z-Compare)
# ============================================================

phase_compare.to_csv("/Users/anokhpalakurthi/Downloads/Belichick_Phase_Comparison.csv", index=False)
comparison.to_csv("/Users/anokhpalakurthi/Downloads/Belichick_vs_League_Yearly.csv", index=False)
phase_z_compare.to_csv("/Users/anokhpalakurthi/Downloads/Belichick_Phase_ZScore_Comparison.csv", index=False)

print("\nExports complete:")
print("- Belichick_Phase_Comparison.csv")
print("- Belichick_vs_League_Yearly.csv")
print("- Belichick_Phase_ZScore_Comparison.csv")

# ============================================================
# 10. SUMMARY INTERPRETATION (Headline Z-Stats)
# ============================================================

phase_z_compare = phase_z_compare.dropna(subset=["Patriots_Z", "League_Z"]).copy()
z_start = phase_z_compare.iloc[0]["Patriots_Z"]
z_end   = phase_z_compare.iloc[-1]["Patriots_Z"]
z_drop  = z_start - z_end

print("\n--- Summary Interpretation (Z-Score Normalized) ---")
print(f"Phase 1 (2000–2005): Patriots averaged {z_start:.2f}σ above league.")
print(f"Phase 4 (2018–2023): Patriots averaged {z_end:.2f}σ relative to league.")
print(f"Net decline: {z_drop:.2f}σ from early dynasty to post-Brady years.\n")

if z_drop > 1:
    print("→ Statistically large decline (>1σ): clear evidence of diminished draft efficiency.")
elif z_drop > 0.5:
    print("→ Moderate decline (~0.5–1σ): evidence of gradual erosion in draft performance.")
else:
    print("→ Minimal standardized decline: largely within expected variance.")

print("\nBelichick’s draft success went from well above league average "
      f"({z_start:.2f}σ) to moderately below it ({z_end:.2f}σ), "
      f"a {z_drop:.2f}σ drop across 24 years.")

# ============================================================
# 11. DRAFT EFFICIENCY RATIO (Z-Score Normalized)
# ============================================================
# Purpose: Compare how efficiently teams convert draft capital into value,
# corrected for year-to-year variance (normalize DrAV and Expected_DrAV by year).

# Z-normalize by draft year
train["Z_DrAV"] = train.groupby("Yr")["DrAV"].transform(lambda x: (x - x.mean()) / x.std())
train["Z_Expected_DrAV"] = train.groupby("Yr")["Expected_DrAV"].transform(lambda x: (x - x.mean()) / x.std())
train["Efficiency_Z"] = train["Z_DrAV"] - train["Z_Expected_DrAV"]

# Team-year efficiency
efficiency_yearly = (
    train.groupby(["Tm", "Yr"], as_index=False)
    .agg(Mean_Efficiency=("Efficiency_Z", "mean"))
)

pats_eff = efficiency_yearly.loc[efficiency_yearly["Tm"] == "NWE", ["Yr", "Mean_Efficiency"]].copy()
league_eff = (
    efficiency_yearly.loc[efficiency_yearly["Tm"] != "NWE"]
    .groupby("Yr", as_index=False)
    .agg(League_Mean_Eff=("Mean_Efficiency", "mean"))
)

eff_compare = pd.merge(pats_eff, league_eff, on="Yr", how="left")

# Visualization — DER over time (Z-normalized)
plt.figure(figsize=(10,6))
sns.lineplot(data=eff_compare, x="Yr", y="Mean_Efficiency", label="Patriots (Belichick Era)", linewidth=2.5)
sns.lineplot(data=eff_compare, x="Yr", y="League_Mean_Eff", label="League Average (Non-Patriots)",
             linestyle="--", color="gray", linewidth=2)
plt.axhline(0, color="black", linestyle="--", lw=1)
plt.title("Draft Efficiency Ratio Over Time (Z-Score Normalized, 2000–2023)")
plt.xlabel("Draft Year")
plt.ylabel("Normalized Efficiency (Z_DrAV – Z_Expected_DrAV)")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# Summary printout
print("\n==================== SECTION 11: DRAFT EFFICIENCY RATIO ====================")
mean_pats_eff   = pats_eff["Mean_Efficiency"].mean()
mean_league_eff = league_eff["League_Mean_Eff"].mean()
eff_gap         = mean_pats_eff - mean_league_eff
best_year_row   = eff_compare.loc[eff_compare["Mean_Efficiency"].idxmax()]
worst_year_row  = eff_compare.loc[eff_compare["Mean_Efficiency"].idxmin()]

print(f"Patriots mean normalized efficiency: {mean_pats_eff:+.3f}")
print(f"League mean normalized efficiency: {mean_league_eff:+.3f}")
print(f"Patriots–League gap (2000–2023): {eff_gap:+.3f}")
print(f"Best year: {int(best_year_row['Yr'])} ({best_year_row['Mean_Efficiency']:+.2f}σ)")
print(f"Worst year: {int(worst_year_row['Yr'])} ({worst_year_row['Mean_Efficiency']:+.2f}σ)")
print("==========================================================================\n")

# ============================================================
# 12. PHASE-LEVEL EFFICIENCY (Patriots vs League, Z-Normalized)
# ============================================================

# Label phases on the normalized rows
train["Phase"] = train["Yr"].apply(assign_phase)

phase_eff = (
    train.groupby(["Phase", "Tm"], as_index=False)
    .agg(Mean_Efficiency=("Efficiency_Z", "mean"))
)

pats_phase_eff = phase_eff.loc[phase_eff["Tm"] == "NWE", ["Phase", "Mean_Efficiency"]].rename(columns={"Mean_Efficiency": "Patriots"})
league_phase_eff = (
    phase_eff.loc[phase_eff["Tm"] != "NWE"]
    .groupby("Phase", as_index=False)
    .agg(League=("Mean_Efficiency", "mean"))
)

phase_eff_compare = pd.merge(pats_phase_eff, league_phase_eff, on="Phase", how="left")
phase_eff_compare["Phase"] = pd.Categorical(phase_eff_compare["Phase"], categories=phase_order, ordered=True)
phase_eff_compare = phase_eff_compare.sort_values("Phase")
phase_eff_compare["Net_Diff"] = phase_eff_compare["Patriots"] - phase_eff_compare["League"]

# Visualization — grouped bars via melt
melt_df = phase_eff_compare.melt(id_vars=["Phase"], value_vars=["Patriots", "League"],
                                 var_name="Group", value_name="Z_Efficiency")

plt.figure(figsize=(10,6))
sns.barplot(data=melt_df, x="Phase", y="Z_Efficiency", hue="Group")
plt.axhline(0, color="black", linestyle="--", lw=1)
plt.title("Average Draft Efficiency (Z-Normalized) by Era Phase (2000–2023)")
plt.xlabel("Belichick Era Phase (6-Year Windows)")
plt.ylabel("Z-Normalized Efficiency")
plt.xticks(rotation=30, ha="right")
plt.legend(title="")
plt.tight_layout()
plt.show()

# Summary printout
print("\n==================== SECTION 12: PHASE-LEVEL EFFICIENCY ====================")
for _, row in phase_eff_compare.iterrows():
    print(f"{row['Phase']}: Patriots {row['Patriots']:+.3f}σ, "
          f"League {row['League']:+.3f}σ, Net diff {row['Net_Diff']:+.3f}σ")

phase_decline = phase_eff_compare.iloc[0]["Patriots"] - phase_eff_compare.iloc[-1]["Patriots"]
print(f"\nTotal decline from Phase 1 to Phase 4: {phase_decline:+.2f}σ")
if abs(phase_decline) > 1:
    print("Interpretation: Strong erosion of draft efficiency (>1σ).")
elif abs(phase_decline) > 0.5:
    print("Interpretation: Moderate decline (~0.5–1σ).")
else:
    print("Interpretation: Marginal change, within noise range.")
print("==========================================================================\n")

# ============================================================
# 13. POSITIONAL ALIGNMENT MATRIX (Z-Normalized)
# ============================================================
# Purpose: Identify which positional groups contributed most to Patriots’ efficiency
# across each era, versus league norms.

# Position consolidation → higher-level groups
train["Pos_Grouped"] = train["Pos"].replace({
    "G": "OL", "T": "OL", "C": "OL", "LS": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL",
    "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
    "WR": "Skill", "TE": "Skill", "RB": "Skill", "FB": "Skill",
    "K": "ST", "P": "ST"
})

# Team-phase efficiency by position group
pos_perf = (
    train.groupby(["Phase", "Pos_Grouped", "Tm"], as_index=False)
    .agg(Mean_Z_Eff=("Efficiency_Z", "mean"))
)

# League baseline per phase/position
league_pos = (
    pos_perf.loc[pos_perf["Tm"] != "NWE"]
    .groupby(["Phase", "Pos_Grouped"], as_index=False)
    .agg(League_Mean=("Mean_Z_Eff", "mean"))
)

# Patriots differential vs league
pats_pos = pos_perf.loc[pos_perf["Tm"] == "NWE", ["Phase", "Pos_Grouped", "Mean_Z_Eff"]].merge(
    league_pos, on=["Phase", "Pos_Grouped"], how="left"
)
pats_pos["Net_Pos_Diff"] = pats_pos["Mean_Z_Eff"] - pats_pos["League_Mean"]

# Heatmap
pos_matrix = pats_pos.pivot(index="Phase", columns="Pos_Grouped", values="Net_Pos_Diff").fillna(0)
pos_matrix = pos_matrix.reindex(index=phase_order)  # ensure chronological rows

plt.figure(figsize=(12,6))
sns.heatmap(pos_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, cbar_kws={"label": "Patriots – League (Z Efficiency)"})
plt.title("Positional Draft Efficiency Alignment Matrix (Z-Normalized, 2000–2023)")
plt.xlabel("Position Group")
plt.ylabel("Belichick Era Phase")
plt.tight_layout()
plt.show()

# Summary & interpretation
print("\n==================== SECTION 13: POSITIONAL ALIGNMENT ====================")
pos_matrix_values = pos_matrix.stack().reset_index()
pos_matrix_values.columns = ["Phase", "Position", "Z_Diff"]

pos_rank = (
    pos_matrix_values.groupby("Position")["Z_Diff"].mean()
    .sort_values(ascending=False)
    .round(2)
)

top_positions = pos_rank.head(5)
bottom_positions = pos_rank.tail(5)

print("\nTop 5 Efficient Draft Positions (Patriots vs League):")
for pos, val in top_positions.items():
    print(f"  {pos}: +{val:.2f}σ")

print("\nBottom 5 Inefficient Draft Positions (Patriots vs League):")
for pos, val in bottom_positions.items():
    print(f"  {pos}: {val:.2f}σ")

avg_z_diff = pos_matrix_values["Z_Diff"].mean()
print(f"\nOverall positional alignment mean: {avg_z_diff:+.2f}σ (Patriots vs League)")

dominant_pos = top_positions.index[0]
weakest_pos  = bottom_positions.index[-1]
print(f"\nInterpretation: The Patriots’ greatest positional edge came at {dominant_pos}, "
      f"while their weakest area was {weakest_pos}. "
      f"Across all groups, Belichick’s drafts averaged {avg_z_diff:+.2f}σ vs league.")
print("==========================================================================\n")

# Optional: CSV of per-phase best/worst positions
phase_summary = (
    pos_matrix_values.groupby(["Phase", "Position"])["Z_Diff"]
    .mean()
    .reset_index()
)
phase_summary_out = []
for phase in phase_summary["Phase"].unique():
    d = phase_summary[phase_summary["Phase"] == phase]
    best = d.loc[d["Z_Diff"].idxmax()]
    worst = d.loc[d["Z_Diff"].idxmin()]
    phase_summary_out.append({
        "Phase": phase,
        "Best_Position": best["Position"], "Best_Z": best["Z_Diff"],
        "Worst_Position": worst["Position"], "Worst_Z": worst["Z_Diff"]
    })
phase_summary_df = pd.DataFrame(phase_summary_out)
phase_summary_df.to_csv("/Users/anokhpalakurthi/Downloads/Belichick_Positional_Phase_Summary.csv", index=False)
print("Exported: Belichick_Positional_Phase_Summary.csv")