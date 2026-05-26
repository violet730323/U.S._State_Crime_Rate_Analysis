import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import textwrap

sns.set_theme(style = "whitegrid", palette = "muted")

base_path = os.path.dirname(__file__)
input_file = os.path.join(base_path, "state_crime_census_2022.csv")

df = pd.read_csv(input_file)

# to be noted: both clearance_rate and num_agencies are 1 for all states
diff_crna = df['clearance_rate'].compare(df['num_agencies'])
if not diff_crna.empty:
    print("There are differences between clearance_rate and num_agencies", diff_crna)
# also total_incidents == cleared_incidents == total_offenses
diff_tici = df['total_incidents'].compare(df['cleared_incidents'])
diff_cito = df['cleared_incidents'].compare(df['total_offenses'])
if not diff_tici.empty or not diff_cito.empty:
    print(f"There are differences between total_incidents and cleared_incidents: {diff_tici} and between total_offenses and cleared_incidents {diff_cito}.")

# flagging DC as an outlier to the rest of the states (it can't be held to
# the same standards)
dc = ["District of Columbia"]

# something is off about the crime rates of some states in terms of how large their
# populations are. After some investigating, this seems to be due to the "NIBRS effect". 
# https://www.themarshallproject.org/2023/07/13/fbi-crime-rates-data-gap-nibrs
df["TI_divided_TP"] = df["total_incidents"] / df["total_population"] * 100
print(df.head())
df_sorted = df.sort_values(by='TI_divided_TP', ascending=False)
# the exceptionally low values of TI_divided_TP shows something odd in the reporting

# finding the cutoff of states that are more than 1 std beneath the mean
mean_rate = df["TI_divided_TP"].mean()
std_rate = df["TI_divided_TP"].std()

cutoff_threshold = mean_rate - (1 * std_rate)
cutoff_df = df[df["TI_divided_TP"] < cutoff_threshold]

print("States statistically likely to have had NIBRS reporting gaps in 2022:")
print(cutoff_df[["NAME", "TI_divided_TP", "poverty_rate_pct"]])

# creating flag label to show which states are full reporters and which ones aren't
df["flag_label"] = "full_report"
df.loc[df["NAME"].isin(cutoff_df["NAME"]), "flag_label"] = "NIBRS_partial"
df.loc[df["NAME"].isin(dc), "flag_label"] = "DC"

# to make it easier to grab each time
flagged_states = cutoff_df["NAME"].tolist()
flagged_states.append("District of Columbia")

df_noflag = df[~df["NAME"].isin(flagged_states)].copy().reset_index(drop=True)

# Grouping features together
socio_economic_features = [
    "median_household_income",
    "unemployment_rate_pct",
    "poverty_rate_pct",
    "bachelors_degree_or_higher_pct",
    "total_population",
    "white_alone_pct",
    "black_alone_pct",
    "hispanic_latino_pct",
    "housing_vacancy_rate_pct",
]

analysis_cols = ["crime_rate_per_100k"] + socio_economic_features
desc_all = df[analysis_cols].describe().round(2)
desc_clean = df_noflag[analysis_cols].describe().round(2)
skew_all = df["crime_rate_per_100k"].skew()
skew_clean = df_noflag["crime_rate_per_100k"].skew()

with open('data_summary.txt', 'w') as f:
    f.write("ALL STATES SUMMARY\n")
    f.write(desc_all.to_string())
    f.write(f"\nCrime skewness: {skew_all:.3f}")
    f.write("\n\nCLEAN STATES SUMMARY\n")
    f.write(desc_clean.to_string())
    f.write(f"\nCrime skewness: {skew_clean:.3f}")

# In the file, we can see that the poverty and unemployment rate remains very similar
# between the all states summary and the clean states summary. This shows that we still
# have a good range of socioeconomic factors (as far as those two are concerned) to work
# with. Furthermore, we can see minimum of crime_rate_per_100k rose from 725.2 to the
# healthier number of 2574.62 after removing the flagged states. We can also see that we
# need to look at the states near the max number of 5971.47 for crime_rate_per_100k since
# it is over 1,500 more than the mean.

labeling_for_graphs = {
    "median_household_income": "Median Household Income ($)",
    "unemployment_rate_pct": "Unemployment Rate (%)",
    "poverty_rate_pct": "Poverty Rate (%)",
    "bachelors_degree_or_higher_pct": "Bachelor's Degree or Higher (%)",
    "total_population": "Total Population",
    "white_alone_pct": "White (Non-Hispanic) (%)",
    "black_alone_pct": "Black / African American (%)",
    "hispanic_latino_pct": "Hispanic / Latino (%)",
    "housing_vacancy_rate_pct": "Housing Vacancy Rate (%)",
}

# Creating a correlation table now for each column against crime_rate_per_100k
def corr_table(data):
    rows = []
    for col in socio_economic_features:
        r_p, p_p = stats.pearsonr(data[col], data["crime_rate_per_100k"])
        r_s, p_s = stats.spearmanr(data[col], data["crime_rate_per_100k"])
        rows.append({
            "Feature":     labeling_for_graphs[col],
            "Pearson r":   round(r_p, 3),
            "Pearson p":   round(p_p, 3),
            "Spearman r":  round(r_s, 3),
            "Spearman p":  round(p_s, 3),
            "P Sig":         "**" if p_p < 0.05 else ("*" if p_p < 0.10 else ""),
            "S Sig":         "**" if p_s < 0.05 else ("*" if p_s < 0.10 else "")
        })
    return pd.DataFrame(rows).sort_values("Pearson r")

corr_clean = corr_table(df_noflag)

print(corr_clean.to_string(index=False))

with open('data_summary.txt', 'a') as f:
    f.write("\n\n Correlation Table\n")
    f.write(corr_clean.to_string(index=False))

# We can see from this table that, interestingly, poverty, income, and unemployment
# aren't strongly correlated to crime rate. At the state level, they don't seem to
# be as strong drivers of crime prediction as one would think. All this really says,
# due to the troubles of getting information in that year, is that a state's poverty
# level didn't reliably predict it's crime rate in 2022. Furthermore, "White
# (Non-Hispanic)" is correlated by the Spearman index. As the percentage of the white
# population increases, the crime rate tends to decrease. Otherwise, the only other
# feature to notice is that states with higher Hispanic populations reported higher
# crime rates in 2022.

# Creating graphs for PPT
# Used Google AI to figure out how to create these graphs
df_plot = corr_clean.sort_values("Pearson r")

# Bar Chart
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(df_plot))
width = 0.35
plt.barh(y_pos - width/2, df_plot["Pearson r"], width, label='Pearson r', color='#3498db')
plt.barh(y_pos + width/2, df_plot["Spearman r"], width, label='Spearman r', color='#e67e22')
for i, row in enumerate(df_plot.itertuples()):
    if row._6: # P Sig
        p_offset = 0.005 if row._2 >= 0 else -0.015
        ha = 'left' if row._2 >= 0 else 'right'
        plt.text(row._2 + p_offset, i - width/2, row._6, 
                 va='center', ha=ha, fontweight='bold', color='#2980b9')
    if row._7: # S Sig
        s_offset = 0.005 if row._4 >= 0 else -0.015
        ha = 'left' if row._4 >= 0 else 'right'
        plt.text(row._4 + s_offset, i + width/2, row._7, 
                 va='center', ha=ha, fontweight='bold', color='#d35400')
plt.yticks(y_pos, df_plot["Feature"])
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Socioeconomic Correlations with Dual Significance\n(* p<0.10, ** p<0.05)", fontsize=14, pad=15)
plt.xlabel("Correlation Coefficient (r)")
plt.legend(loc='lower right')
plt.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig("barchart_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# Standard Heatmap
annot_matrix = df_plot.apply(
    lambda x: [f"{x['Pearson r']:.3f}{x['P Sig']}", f"{x['Spearman r']:.3f}{x['S Sig']}"], 
    axis=1, result_type='expand'
)
annot_matrix.index = df_plot["Feature"]
annot_matrix.columns = ["Pearson r", "Spearman r"]
heatmap_values = df_plot.set_index("Feature")[["Pearson r", "Spearman r"]]
plt.figure(figsize=(10, 7))
sns.heatmap(
    heatmap_values, 
    annot=annot_matrix, 
    fmt="", 
    cmap="RdBu_r", 
    center=0, 
    linewidths=.5,
    annot_kws={"size": 11, "weight": "bold"}
)
plt.title("Socioeconomic Correlation Heatmap", fontsize=14, pad=15)
plt.ylabel("")
plt.tight_layout()
plt.savefig("heatmap_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# scatter grids for each socioeconomic feature against crime_rate_per_100k
# the regression line and r/p values are computed based on the clean states only
# shows which features move with crime and in which direction
state_abbr = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}
df["abbr"] = df["NAME"].map(state_abbr)

color_categories = {
    "full_report":    "#4a90d9",
    "NIBRS_partial":  "#e67e22",
    "DC":             "#c0392b",
}

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.flatten()

for i, feat in enumerate(socio_economic_features):
    ax = axes[i]
 
    # Flagged states (faded, for reference)
    for flag, grp in df.groupby("flag_label"):
        alpha  = 0.75 if flag == "full_report" else 0.35
        colour = color_categories[flag]
        ax.scatter(grp[feat], grp["crime_rate_per_100k"],
                   color=colour, alpha=alpha, s=35, zorder=3)
 
    # State abbreviation labels
    for _, row in df.iterrows():
        ax.annotate(row["abbr"], (row[feat], row["crime_rate_per_100k"]),
                    fontsize=4.5, ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points", color="grey")
 
    # Regression line on clean states only
    sub = df_noflag[[feat, "crime_rate_per_100k"]].dropna()
    m, b, r, p, _ = stats.linregress(sub[feat], sub["crime_rate_per_100k"])
    xr = np.linspace(sub[feat].min(), sub[feat].max(), 100)
    ax.plot(xr, m * xr + b, color="steelblue", linewidth=1.5, linestyle="--")
 
    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
    ax.set_title(f"{labeling_for_graphs[feat]}\nr={r:+.2f}{sig}, p={p:.2f}",
                 fontsize=8, fontweight="bold")
    ax.set_ylabel("Crime Rate /100k", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    if "population" in feat:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
 
plt.suptitle(
    "Socioeconomic Predictors vs Crime Rate per 100k (2022)\n"
    "Blue = Full Reported Crime  ·  Orange = Partial Crime Reported  ·  Red = DC",
    fontsize=11, fontweight="bold",
)
plt.savefig("scatterplot_allpredictors.png", dpi=150, bbox_inches="tight")
plt.close()

# Another correlation graph
corr_matrix = df_noflag[socio_economic_features].corr()
labels = [labeling_for_graphs[f] for f in socio_economic_features]
wrapped_labels = [textwrap.fill(label, width=15) for label in labels]
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    xticklabels=wrapped_labels, yticklabels=labels,
    linewidths=0.4,
    annot_kws={"size": 8},
)
plt.xticks(rotation=0, ha="center", fontsize=8)
plt.yticks(fontsize=8)
plt.title(
    "Inter-Correlation Heatmap\n",
    fontsize=10, fontweight="bold",
)
plt.tight_layout()
plt.savefig("standard_correlation.png", dpi=150, bbox_inches="tight")
plt.close()