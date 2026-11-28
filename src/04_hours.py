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


hours_band = [0, 5, 10, 25, 50, 100, 200, 500, np.inf]

labels = [
    "0-5",
    "5-10",
    "10-25",
    "25-50",
    "50-100",
    "100-200",
    "200-500",
    "500+"
]

num_rev = df["hours_played"].count()

print(df["hours_played"].count())


bad_reviews = (df["not_recommended"] == 1).sum()

percent_of_bad_reviews = bad_reviews / num_rev * 100

print(percent_of_bad_reviews)

df["hours_band"] = pd.cut(
    df["hours_played"],
    bins=hours_band,
    right=False
)

df.groupby("hours_band", observed=True)["not_recommended"].sum()


bad_reviews_by_band = df.groupby(
    "hours_band", observed=True
)["not_recommended"].sum()

total_reviews_by_band = df.groupby(
    "hours_band", observed=True
)["not_recommended"].count()


good_reviews_by_band = total_reviews_by_band - bad_reviews_by_band


print(bad_reviews_by_band)

print(total_reviews_by_band) 

print(good_reviews_by_band)

percent_by_band = bad_reviews_by_band / total_reviews_by_band * 100


band_stats = pd.DataFrame({
    "total_reviews": total_reviews_by_band,
    "bad_reviews": bad_reviews_by_band,
    "good_reviews": good_reviews_by_band,
    "percent_bad_reviews": percent_by_band
})

band_stats.to_csv("results/hours_table.csv")

