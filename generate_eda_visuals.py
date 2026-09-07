import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set clean publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CBD5E0'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('figures', exist_ok=True)

# Load dataset
df = pd.read_csv('data/merged_news_pld_cmo_by_region_date_clean.csv')
df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)
df_scope = df[(df['Date_parsed'] >= '2018-01-01') & (df['Date_parsed'] <= '2026-08-01')].copy()

# -----------------------------------------------------------------------------
# 1. EDA Regional Distribution (eda_regional_distribution.png)
# -----------------------------------------------------------------------------
print("1. Generating eda_regional_distribution.png...")
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=250)

regions = ['Southeast', 'Central-West', 'South', 'North', 'Northeast', 'National']
counts = [len(df[df['Region'] == r]) for r in regions]
colors = ['#1B365D', '#2C5282', '#3182CE', '#4299E1', '#63B3ED', '#A0AEC0']

bars = ax.bar(regions, counts, color=colors, width=0.55, edgecolor='black', linewidth=0.7)
ax.set_ylabel('Number of Observations', fontsize=9.5, fontweight='bold', color='#1B365D')
ax.set_title('Regional Distribution of Observations Across Brazilian Submarkets (2018–2026)', fontsize=10.5, fontweight='bold', color='#1B365D', pad=10)
ax.set_ylim(0, 3600)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.annotate(f"{h:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8.0, fontweight='bold', color='#2D3748')

plt.tight_layout()
fig.savefig('figures/eda_regional_distribution.png', dpi=250)
plt.close(fig)
print("Saved figures/eda_regional_distribution.png")

# -----------------------------------------------------------------------------
# 2. EDA Yearly Distribution (eda_yearly_distribution.png)
# -----------------------------------------------------------------------------
print("2. Generating eda_yearly_distribution.png...")
fig, ax1 = plt.subplots(figsize=(8.0, 3.8), dpi=250)

years = list(range(2018, 2027))
mkt_by_yr = [len(df_scope[df_scope['Date_parsed'].dt.year == y]) for y in years]

# Unique dates for news articles
df_unique = df_scope.drop_duplicates(subset=['Date_parsed'])
news_by_yr = [df_unique[df_unique['Date_parsed'].dt.year == y]['Articles'].sum() for y in years]

x = np.arange(len(years))
w = 0.38

rects1 = ax1.bar(x - w/2, mkt_by_yr, width=w, label='Market Observations (Regional Records)', color='#1B365D', edgecolor='black', linewidth=0.6)
ax1.set_ylabel('Market Observations Count', fontsize=9.0, fontweight='bold', color='#1B365D')
ax1.set_xticks(x)
ax1.set_xticklabels([str(y) for y in years], fontsize=8.5, fontweight='bold')
ax1.set_ylim(0, 2600)
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

# Secondary axis for news articles
ax2 = ax1.twinx()
rects2 = ax2.bar(x + w/2, news_by_yr, width=w, label='Energy News Articles (Unique Dates)', color='#D97706', edgecolor='black', linewidth=0.6)
ax2.set_ylabel('News Articles Published', fontsize=9.0, fontweight='bold', color='#D97706')
ax2.set_ylim(0, 2200)
ax2.grid(False)

# Title & combined legend
ax1.set_title('Year-Wise Distribution of Market Observations & Energy News Records (2018–2026)', fontsize=10.0, fontweight='bold', color='#1B365D', pad=10)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.0, frameon=True)

# Annotations
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:,}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=7.0, color='#1B365D', fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax2.annotate(f"{h:,}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=7.0, color='#B45309', fontweight='bold')

plt.tight_layout()
fig.savefig('figures/eda_yearly_distribution.png', dpi=250)
plt.close(fig)
print("Saved figures/eda_yearly_distribution.png")

# -----------------------------------------------------------------------------
# 3. EDA News Topic Distribution (eda_news_topic_distribution.png)
# -----------------------------------------------------------------------------
print("3. Generating eda_news_topic_distribution.png...")
fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=250)

topics = [
    ('Rainfall Uncertainty', 'future rainfall uncertainty', '#3B82F6'),
    ('Flood Mentions', 'Flood', '#06B6D4'),
    ('El Niño Patterns', 'El nino', '#8B5CF6'),
    ('Drought Events', 'Drought', '#EF4444'),
    ('La Niña Patterns', 'La nina', '#10B981'),
    ('Thermal Fuel Costs', 'thermal fuel costs', '#F59E0B'),
    ('Hydro Reservoir Levels', 'hydro reservoir levels', '#6366F1'),
    ('Renewable Curtailment', 'Curtailment', '#14B8A6')
]

labels_clean = [t[0] for t in topics]
counts_topic = [df_unique[t[1]].sum() for t in topics]
colors_topic = [t[2] for t in topics]
total_mentions = sum(counts_topic)

y_pos = np.arange(len(labels_clean))
bars = ax.barh(y_pos, counts_topic, color=colors_topic, edgecolor='black', linewidth=0.6, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_clean, fontsize=8.5, fontweight='bold')
ax.invert_yaxis()  # top-down
ax.set_xlabel('Total Article Mentions (2018–2026 Scope)', fontsize=9.0, fontweight='bold', color='#1B365D')
ax.set_title('Distribution of Energy News Mentions Across Domain-Specific Topics', fontsize=10.0, fontweight='bold', color='#1B365D', pad=10)
ax.set_xlim(0, 2900)
ax.grid(True, linestyle='--', alpha=0.5, axis='x')

for bar in bars:
    w_val = bar.get_width()
    pct = (w_val / total_mentions) * 100
    ax.annotate(f"{int(w_val):,}  ({pct:.1f}%)",
                xy=(w_val, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points", va='center', fontsize=7.8, fontweight='bold', color='#2D3748')

plt.tight_layout()
fig.savefig('figures/eda_news_topic_distribution.png', dpi=250)
plt.close(fig)
print("Saved figures/eda_news_topic_distribution.png")

# -----------------------------------------------------------------------------
# 4. EDA News Topics Over Time (eda_news_topics_over_time.png)
# -----------------------------------------------------------------------------
print("4. Generating eda_news_topics_over_time.png...")
fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=250)

df_unique['YearMonth'] = df_unique['Date_parsed'].dt.to_period('M')
monthly_topics = df_unique.groupby('YearMonth').agg({
    'Drought': 'sum',
    'hydro reservoir levels': 'sum',
    'Flood': 'sum',
    'Curtailment': 'sum',
    'future rainfall uncertainty': 'sum'
}).reset_index()
monthly_topics['Date'] = monthly_topics['YearMonth'].dt.to_timestamp()

dates_m = monthly_topics['Date']
y_stack = [
    monthly_topics['Drought'],
    monthly_topics['hydro reservoir levels'],
    monthly_topics['Flood'],
    monthly_topics['Curtailment'],
    monthly_topics['future rainfall uncertainty']
]
labels_stack = ['Drought', 'Hydro Reservoir Levels', 'Flood', 'Renewable Curtailment', 'Rainfall Uncertainty']
colors_stack = ['#EF4444', '#F59E0B', '#06B6D4', '#10B981', '#3B82F6']

ax.stackplot(dates_m, y_stack, labels=labels_stack, colors=colors_stack, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2021-12-31'), color='#FFCCCC', alpha=0.4, label='2020–2021 Water Crisis')

ax.set_ylabel('Monthly Mentions (Stacked Count)', fontsize=9.0, fontweight='bold', color='#1B365D')
ax.set_title('Temporal Evolution of Energy News Topics Across the Research Timeline (2018–2026)', fontsize=10.0, fontweight='bold', color='#1B365D', pad=10)
ax.set_xlim(pd.to_datetime('2018-01-01'), pd.to_datetime('2026-07-01'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=8.0, frameon=True, ncol=2)

plt.tight_layout()
fig.savefig('figures/eda_news_topics_over_time.png', dpi=250)
plt.close(fig)
print("Saved figures/eda_news_topics_over_time.png")

# -----------------------------------------------------------------------------
# 5. EDA News Volume & Sentiment (eda_news_volume_sentiment.png)
# -----------------------------------------------------------------------------
print("5. Generating eda_news_volume_sentiment.png...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 4.4), dpi=250, sharex=True, gridspec_kw={'height_ratios': [1.3, 1.0]})

monthly_sent = df_unique.groupby('YearMonth').agg({
    'Articles': 'sum',
    'Avg Sentiment': 'mean'
}).reset_index()
monthly_sent['Date'] = monthly_sent['YearMonth'].dt.to_timestamp()

# Top Panel: News Article Volume
bars_vol = ax1.bar(monthly_sent['Date'], monthly_sent['Articles'], width=24, color='#2563EB', alpha=0.8, edgecolor='#1E40AF', linewidth=0.4, label='Monthly Article Volume')
ax1.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2021-12-31'), color='#FFCCCC', alpha=0.35, label='2020–2021 Crisis Window')
ax1.set_ylabel('Articles / Month', fontsize=8.5, fontweight='bold', color='#1B365D')
ax1.set_title('Energy News Article Volume & Sentiment Polarity Trends (2018–2026)', fontsize=10.0, fontweight='bold', color='#1B365D', pad=8)
ax1.legend(loc='upper left', fontsize=7.8, frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# Bottom Panel: Average Lexical Sentiment
sent_vals = monthly_sent['Avg Sentiment'].fillna(0)
ax2.plot(monthly_sent['Date'], sent_vals, color='#7C3AED', linewidth=1.5, label='Mean Sentiment Score')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2021-12-31'), color='#FFCCCC', alpha=0.35)
ax2.set_ylabel('Sentiment Score', fontsize=8.5, fontweight='bold', color='#1B365D')
ax2.set_xlabel('Timeline (Year)', fontsize=9.0, fontweight='bold', color='#1B365D')
ax2.set_ylim(-0.45, 0.45)
ax2.legend(loc='lower left', fontsize=7.8, frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlim(pd.to_datetime('2018-01-01'), pd.to_datetime('2026-07-01'))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
fig.savefig('figures/eda_news_volume_sentiment.png', dpi=250)
plt.close(fig)
print("Saved figures/eda_news_volume_sentiment.png")
print("All 5 EDA visuals generated successfully!")
