"""
============================================================
  Superstore Sales — End-to-End Data Analysis Project
  Author  : Mahendra Meena 
  Date    : 06/04/2026
  Dataset : Sample Superstore (9,994 transactions, 2014-2017)
============================================================
"""

# ── 0. IMPORTS & CONFIGURATION ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

sns.set_context("notebook", font_scale=1.15)
plt.style.use("ggplot")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

OUTPUT_DIR = "superstore_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

# ── 1. DATA GENERATION (mirrors the real Superstore dataset) ─────────────────
print("\n" + "="*60)
print("  STEP 1 — Generating synthetic Superstore dataset")
print("="*60)

np.random.seed(42)
N = 9994

CATEGORIES = {
    "Furniture":       ["Chairs", "Tables", "Bookcases", "Furnishings"],
    "Technology":      ["Phones", "Accessories", "Machines", "Copiers"],
    "Office Supplies": ["Storage", "Binders", "Art", "Labels", "Fasteners", "Supplies", "Envelopes", "Paper"],
}
REGIONS     = ["West", "East", "Central", "South"]
STATES_MAP  = {
    "West":    ["California", "Washington", "Oregon", "Nevada", "Arizona"],
    "East":    ["New York", "Pennsylvania", "Ohio", "Virginia", "Georgia"],
    "Central": ["Texas", "Illinois", "Michigan", "Missouri", "Indiana"],
    "South":   ["Florida", "North Carolina", "Tennessee", "Alabama", "Mississippi"],
}
SEGMENTS    = ["Consumer", "Corporate", "Home Office"]
SHIP_MODES  = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_DAYS   = {"Standard Class": (3,7), "Second Class": (2,5), "First Class": (1,3), "Same Day": (0,1)}

# Build category/sub-category arrays
cat_list, sub_list = [], []
for cat, subs in CATEGORIES.items():
    for sub in subs:
        cat_list.append(cat)
        sub_list.append(sub)
cat_arr = np.array(cat_list)
sub_arr = np.array(sub_list)

# Profit margin profiles per sub-category
MARGIN = {
    "Chairs": 0.12,   "Tables": -0.085, "Bookcases": -0.04, "Furnishings": 0.09,
    "Phones": 0.15,   "Accessories": 0.18, "Machines": 0.08, "Copiers": 0.25,
    "Storage": 0.13,  "Binders": 0.14, "Art": 0.20, "Labels": 0.19,
    "Fasteners": 0.22,"Supplies": 0.10, "Envelopes": 0.17, "Paper": 0.21,
}

idx      = np.random.randint(0, len(cat_arr), N)
category = cat_arr[idx]
sub_cat  = sub_arr[idx]

segment  = np.random.choice(SEGMENTS,   N, p=[0.52, 0.30, 0.18])
region   = np.random.choice(REGIONS,    N, p=[0.32, 0.30, 0.21, 0.17])
state    = np.array([np.random.choice(STATES_MAP[r]) for r in region])
ship_mode= np.random.choice(SHIP_MODES, N, p=[0.60, 0.19, 0.15, 0.06])

order_dates = pd.to_datetime("2014-01-01") + pd.to_timedelta(
    np.random.randint(0, 4*365, N), unit="D"
)
ship_delays = np.array([
    np.random.randint(*SHIP_DAYS[s]) for s in ship_mode
])
ship_dates  = order_dates + pd.to_timedelta(ship_delays, unit="D")

# Sales: right-skewed
sales_base = np.random.lognormal(mean=4.5, sigma=1.3, size=N).clip(0.44, 22638)
quantity   = np.random.randint(1, 14, N)

# Discounts: clustered at 0, 0.2, 0.3, 0.5, 0.7
disc_vals  = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
disc_probs = np.array([0.40, 0.05, 0.22, 0.12, 0.05, 0.08, 0.03, 0.04, 0.01])
discount   = np.random.choice(disc_vals, N, p=disc_probs)

sales      = sales_base * quantity * (1 - discount * 0.5)

# Profit: margin + discount erosion + noise
base_margin= np.array([MARGIN[s] for s in sub_cat])
eff_margin = base_margin - discount * 0.6 + np.random.normal(0, 0.04, N)
profit     = sales * eff_margin
profit     = profit.clip(-6600, 8400)

order_ids  = ["CA-" + str(2014 + (order_dates[i].year - 2014)) + "-" + str(100000 + i)
              for i in range(N)]

df = pd.DataFrame({
    "Row ID":      range(1, N+1),
    "Order ID":    order_ids,
    "Order Date":  order_dates,
    "Ship Date":   ship_dates,
    "Ship Mode":   ship_mode,
    "Segment":     segment,
    "State":       state,
    "Region":      region,
    "Category":    category,
    "Sub-Category":sub_cat,
    "Sales":       sales.round(2),
    "Quantity":    quantity,
    "Discount":    discount,
    "Profit":      profit.round(2),
})

# Inject a few duplicates and nulls to mimic real data
dup_rows = df.sample(6, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)
df.loc[df.sample(4, random_state=2).index, "Sales"] = np.nan

df.to_csv("SampleSuperstore.csv", index=False, encoding="windows-1252")
print(f"  Generated dataset: {df.shape[0]} rows × {df.shape[1]} cols")


# ── 2. INGEST & STRUCTURAL AUDIT ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Data Ingestion & Structural Audit")
print("="*60)

df = pd.read_csv("SampleSuperstore.csv", encoding="windows-1252")
print(f"\n  Shape  : {df.shape}")
print(f"\n  dtypes :\n{df.dtypes}")
print(f"\n  Nulls  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n  Dupes  : {df.duplicated().sum()}")


# ── 3. DATA CLEANING ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Data Cleaning & Standardisation")
print("="*60)

# 3a. Remove duplicates
dupe_count = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"  Removed {dupe_count} duplicate rows  → {len(df)} rows remaining")

# 3b. Drop Row ID
if "Row ID" in df.columns:
    df.drop("Row ID", axis=1, inplace=True)
    print("  Dropped 'Row ID' column")

# 3c. Type casting
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"]  = pd.to_datetime(df["Ship Date"])

# 3d. Temporal engineering
df["Order Year"]      = df["Order Date"].dt.year
df["Order Month"]     = df["Order Date"].dt.month
df["Order MonthName"] = df["Order Date"].dt.strftime("%b")
df["Shipping Days"]   = (df["Ship Date"] - df["Order Date"]).dt.days

# 3e. Impute missing Sales with median
null_count = df["Sales"].isnull().sum()
df["Sales"].fillna(df["Sales"].median(), inplace=True)
print(f"  Imputed {null_count} missing Sales values with median")

# 3f. Derived KPIs
df["Profit Margin %"] = (df["Profit"] / df["Sales"] * 100).round(2)

print(f"\n  Final clean shape: {df.shape}")
print(f"\n  Sample:\n{df[['Order Date','Category','Sub-Category','Sales','Profit','Discount']].head(4).to_string()}")


# ── 4. DESCRIPTIVE STATISTICS ────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Descriptive Statistics")
print("="*60)

print("\n", df[["Sales","Profit","Quantity","Discount"]].describe().round(2).to_string())

skew = df[["Sales","Profit"]].skew()
print(f"\n  Skewness → Sales: {skew['Sales']:.2f}  |  Profit: {skew['Profit']:.2f}")


# ── 5. EDA VISUALISATIONS ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — Exploratory Data Analysis")
print("="*60)

# ── Fig 1: Category Sales vs Profit ─────────────────────────────────────────
cat_summary = df.groupby("Category")[["Sales","Profit"]].sum().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Category Performance: Sales vs Profit", fontweight="bold", fontsize=14)

colors = ["#4C72B0", "#DD8452", "#55A868"]
axes[0].bar(cat_summary["Category"], cat_summary["Sales"] / 1e6, color=colors)
axes[0].set_title("Total Sales by Category")
axes[0].set_ylabel("Sales ($M)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

axes[1].bar(cat_summary["Category"], cat_summary["Profit"] / 1e3, color=colors)
axes[1].set_title("Total Profit by Category")
axes[1].set_ylabel("Profit ($K)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))

plt.tight_layout()
save(fig, "01_category_sales_profit.png")


# ── Fig 2: Sub-category profit margin (Furniture Paradox) ───────────────────
sub_margin = (
    df.groupby(["Category","Sub-Category"])[["Sales","Profit"]]
    .sum()
    .assign(**{"Margin %": lambda x: (x["Profit"] / x["Sales"] * 100)})
    .reset_index()
    .sort_values("Margin %")
)

fig, ax = plt.subplots(figsize=(11, 7))
bar_colors = ["#d62728" if m < 0 else "#2ca02c" for m in sub_margin["Margin %"]]
bars = ax.barh(sub_margin["Sub-Category"], sub_margin["Margin %"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Profit Margin (%)")
ax.set_title("Sub-Category Profit Margins  |  Red = Loss-Making", fontweight="bold")
for bar, val in zip(bars, sub_margin["Margin %"]):
    ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
save(fig, "02_subcategory_margins.png")


# ── Fig 3: Discount vs Profit — the Profit Cliff ────────────────────────────
disc_profit = df.groupby("Discount")["Profit"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(disc_profit["Discount"], disc_profit["Profit"], marker="o", color="#4C72B0", lw=2)
ax.axhline(0, color="red", linestyle="--", lw=1.2, label="Break-even")
ax.axvline(0.30, color="orange", linestyle="--", lw=1.2, label="30% cliff")
ax.fill_between(disc_profit["Discount"], disc_profit["Profit"], 0,
                where=disc_profit["Profit"] < 0, alpha=0.15, color="red", label="Loss zone")
ax.set_xlabel("Discount Rate")
ax.set_ylabel("Average Profit ($)")
ax.set_title("The Discount–Profit Relationship  |  'Profit Cliff' at 30%", fontweight="bold")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend()
plt.tight_layout()
save(fig, "03_discount_profit_cliff.png")


# ── Fig 4: Regional performance ─────────────────────────────────────────────
region_summary = df.groupby("Region")[["Sales","Profit"]].sum().reset_index()
region_summary["Margin %"] = region_summary["Profit"] / region_summary["Sales"] * 100

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Regional Performance Dashboard", fontweight="bold", fontsize=14)

rcols = ["#4C72B0","#DD8452","#55A868","#C44E52"]
axes[0].bar(region_summary["Region"], region_summary["Sales"]/1e6, color=rcols)
axes[0].set_title("Sales ($M)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.1f}M"))

axes[1].bar(region_summary["Region"], region_summary["Profit"]/1e3, color=rcols)
axes[1].set_title("Profit ($K)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))

axes[2].bar(region_summary["Region"], region_summary["Margin %"], color=rcols)
axes[2].set_title("Profit Margin (%)")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))

plt.tight_layout()
save(fig, "04_regional_performance.png")


# ── Fig 5: Profit distribution boxplot by Category ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="Category", y="Profit", data=df, palette="Set2", ax=ax,
            flierprops=dict(marker="o", markersize=3, alpha=0.4))
ax.set_ylim(-500, 500)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.set_title("Profit Distribution by Category  |  Zoomed to IQR", fontweight="bold")
ax.set_ylabel("Profit ($)")
plt.tight_layout()
save(fig, "05_profit_boxplot.png")


# ── Fig 6: Monthly sales time-series ────────────────────────────────────────
monthly = df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
monthly["Year"] = monthly["Order Date"].dt.year

fig, ax = plt.subplots(figsize=(14, 6))
for yr, grp in monthly.groupby("Year"):
    ax.plot(grp["Order Date"], grp["Sales"]/1e3, marker=".", label=str(yr), lw=1.8)
ax.set_title("Monthly Sales Performance  |  2014–2017", fontweight="bold")
ax.set_ylabel("Sales ($K)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax.legend(title="Year")
plt.tight_layout()
save(fig, "06_monthly_sales_timeseries.png")


# ── Fig 7: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[["Sales","Profit","Quantity","Discount","Shipping Days","Profit Margin %"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap of Key Variables", fontweight="bold")
plt.tight_layout()
save(fig, "07_correlation_heatmap.png")


# ── Fig 8: Segment analysis ──────────────────────────────────────────────────
seg = df.groupby("Segment")[["Sales","Profit"]].sum()
seg["Margin %"] = seg["Profit"] / seg["Sales"] * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Customer Segment Analysis", fontweight="bold", fontsize=14)
seg_colors = ["#4C72B0","#DD8452","#55A868"]

axes[0].pie(seg["Sales"], labels=seg.index, autopct="%1.1f%%",
            colors=seg_colors, startangle=140)
axes[0].set_title("Sales Share by Segment")

axes[1].bar(seg.index, seg["Margin %"], color=seg_colors)
axes[1].set_title("Profit Margin % by Segment")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))

plt.tight_layout()
save(fig, "08_segment_analysis.png")


# ── Fig 9: Ship mode & SLA analysis ─────────────────────────────────────────
ship = df.groupby("Ship Mode")["Shipping Days"].mean().reset_index().sort_values("Shipping Days")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Shipping Mode & Lead Time Analysis", fontweight="bold", fontsize=14)

axes[0].barh(ship["Ship Mode"], ship["Shipping Days"], color="#4C72B0")
axes[0].set_xlabel("Avg Shipping Days")
axes[0].set_title("Average Lead Time by Ship Mode")

sla_check = df[df["Ship Mode"] == "Same Day"]["Shipping Days"].value_counts().sort_index()
axes[1].bar(sla_check.index.astype(str), sla_check.values, color=["#2ca02c","#d62728","#ff7f0e"])
axes[1].set_title("'Same Day' — Actual Days to Ship\n(0 = SLA met; 1+ = SLA failure)")
axes[1].set_xlabel("Days to Ship")
axes[1].set_ylabel("Order Count")

plt.tight_layout()
save(fig, "09_shipping_sla_analysis.png")


# ── Fig 10: Top & bottom states by profit ───────────────────────────────────
state_profit = df.groupby("State")["Profit"].sum().sort_values()
bottom5 = state_profit.head(5)
top5    = state_profit.tail(5)
combined = pd.concat([bottom5, top5])

fig, ax = plt.subplots(figsize=(11, 6))
colors_sb = ["#d62728"]*5 + ["#2ca02c"]*5
ax.barh(combined.index, combined.values / 1e3, color=colors_sb)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Total Profit ($K)")
ax.set_title("Top 5 and Bottom 5 States by Profit  |  Identifying Loss States", fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
plt.tight_layout()
save(fig, "10_top_bottom_states.png")


# ── 6. KEY BUSINESS INSIGHTS SUMMARY ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — Key Business Insights")
print("="*60)

total_sales  = df["Sales"].sum()
total_profit = df["Profit"].sum()
overall_margin = total_profit / total_sales * 100

print(f"\n  Total Sales  : ${total_sales:>12,.0f}")
print(f"  Total Profit : ${total_profit:>12,.0f}")
print(f"  Overall Margin: {overall_margin:.2f}%")

print("\n  Category breakdown:")
print(df.groupby("Category")[["Sales","Profit"]].sum()
      .assign(**{"Margin%": lambda x: (x.Profit/x.Sales*100).round(1)})
      .to_string())

high_disc = df[df["Discount"] >= 0.30]
print(f"\n  Orders with ≥30% discount: {len(high_disc):,}  ({len(high_disc)/len(df)*100:.1f}%)")
print(f"  Avg profit on those orders: ${high_disc['Profit'].mean():.2f}")
print(f"  Avg profit on 0% discount : ${df[df['Discount']==0]['Profit'].mean():.2f}")

loss_subcats = (df.groupby("Sub-Category")[["Sales","Profit"]].sum()
                .assign(**{"Margin%": lambda x: (x.Profit/x.Sales*100).round(1)})
                .query("`Margin%` < 0"))
print(f"\n  Loss-making sub-categories:\n{loss_subcats.to_string()}")

print("\n" + "="*60)
print("  COMPLETE — All charts saved to ./superstore_outputs/")
print("="*60)
