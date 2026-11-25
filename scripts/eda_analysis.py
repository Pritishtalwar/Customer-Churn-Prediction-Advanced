from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
OUTPUT = ROOT / "eda"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
sns.set_theme(style="whitegrid")

missing = df.isna().mean().mul(100).sort_values(ascending=False)
missing.to_csv(OUTPUT / "missing_percentages.csv")
df.nunique(dropna=True).sort_values().to_frame("unique_values").to_csv(
    OUTPUT / "unique_values.csv"
)

fig, ax = plt.subplots(figsize=(7, 5))
df["Churn"].value_counts().sort_index().plot(
    kind="bar",
    ax=ax,
    color=["#94a3b8", "#ef4444"],
)
ax.set_xticklabels(["Stay", "Churn"], rotation=0)
ax.set_title("Churn Distribution")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.savefig(OUTPUT / "churn_distribution.png", dpi=160, bbox_inches="tight")
plt.close()

numeric = df.select_dtypes(include=[np.number]).columns.drop(
    ["Churn", "CustomerID"],
    errors="ignore",
)
categorical = df.select_dtypes(exclude=[np.number]).columns

for column in numeric:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=df, x=column, hue="Churn", bins=40, kde=True, ax=axes[0])
    axes[0].set_title(f"{column} Distribution")
    sns.boxplot(data=df, x="Churn", y=column, ax=axes[1])
    axes[1].set_title(f"{column} by Churn")
    plt.tight_layout()
    safe_name = column.lower().replace(" ", "_")
    plt.savefig(OUTPUT / f"{safe_name}_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

for column in categorical:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[column].value_counts().head(15).plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"{column} Counts")
    axes[0].tick_params(axis="x", rotation=45)
    pd.crosstab(df[column], df["Churn"], normalize="index").plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
    )
    axes[1].set_title(f"{column} Churn Rate")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    safe_name = column.lower().replace(" ", "_")
    plt.savefig(OUTPUT / f"{safe_name}_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

correlations = df[numeric.tolist() + ["Churn"]].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=False, cmap="coolwarm", center=0)
plt.title("Numerical Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()

print("EDA complete: eda")
