import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

shap_df = pd.read_csv("results/shap_values.csv")
df = pd.read_csv("data/clean.csv")

print(shap_df.head())
print(shap_df.columns)

print(df.head())
print(df.columns)

plt.figure(figsize=(10,6))

plt.scatter(
    shap_df["hours_played"],
    shap_df["shap_hours_played"],
    alpha=0.7,
    s=8
)

plt.xscale("log")
plt.axhline(y=0, color="black", linestyle="--")

plt.xlabel("Hours played")
plt.ylabel("SHAP Value of Hours Played")
plt.title("Effect of Playtime on Negative Review Predictions")

plt.tight_layout()
plt.savefig("figures/hours_shap.png", dpi=300)
plt.show()

