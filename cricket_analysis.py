# %% Libraries and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid', palette='Dark2')

print("Libraries loaded")

# %% Load data
deliveries = pd.read_csv('data/deliveries.csv')
matches = pd.read_csv('data/matches.csv')

print("Deliveries shape:", deliveries.shape)
print("Matches shape:", matches.shape)
print("\n--- Deliveries columns ---")
print(deliveries.columns.tolist())
print("\n--- Matches columns ---")
print(matches.columns.tolist())

# %% Clean and merge data
matches = matches.rename(columns={'id': 'match_id'})

df = deliveries.merge(matches[['match_id', 'season', 'venue', 'city', 'winner', 'date']], on='match_id', how='left')

# Fix data types
df['season'] = df['season'].astype(str)
df['date'] = pd.to_datetime(df['date'])

# Check nulls
print("Nulls in deliveries:")
print(deliveries.isnull().sum()[deliveries.isnull().sum() > 0])
print("\nNulls in matches:")
print(matches.isnull().sum()[matches.isnull().sum() > 0])

print("\nMerged shape:", df.shape)
print("Seasons available:", sorted(df['season'].unique()))
print("Sample row:")
df.head(2)

# %% Feature Engineering — Batting stats
batting = df.groupby('batter').agg(
    total_runs    = ('batsman_runs', 'sum'),
    balls_faced   = ('ball', 'count'),
    innings       = ('match_id', 'nunique'),
    dismissals    = ('is_wicket', 'sum'),
    fours         = ('batsman_runs', lambda x: (x == 4).sum()),
    sixes         = ('batsman_runs', lambda x: (x == 6).sum()),
).reset_index()

batting['strike_rate']   = (batting['total_runs'] / batting['balls_faced'] * 100).round(2)
batting['average']       = (batting['total_runs'] / batting['dismissals'].replace(0, 1)).round(2)
batting['boundary_pct']  = ((batting['fours'] + batting['sixes']) / batting['balls_faced'] * 100).round(2)
batting['consistency'] = batting['batter'].map(df.groupby('batter')['batsman_runs'].std())

# Filter out players with too few innings (noise)
batting = batting[batting['innings'] >= 20].reset_index(drop=True)

print("Batting stats shape:", batting.shape)
print(batting[['batter','total_runs','strike_rate','average','boundary_pct']].sort_values('total_runs', ascending=False).head(10).to_string())

# %% Feature Engineering — Bowling stats + Venue effects
bowling = df.groupby('bowler').agg(
    runs_conceded = ('total_runs', 'sum'),
    balls_bowled  = ('ball', 'count'),
    wickets       = ('is_wicket', 'sum'),
    matches       = ('match_id', 'nunique'),
).reset_index()

bowling['overs']        = (bowling['balls_bowled'] / 6).round(2)
bowling['economy']      = (bowling['runs_conceded'] / bowling['overs']).round(2)
bowling['strike_rate']  = (bowling['balls_bowled'] / bowling['wickets'].replace(0, 1)).round(2)
bowling['average']      = (bowling['runs_conceded'] / bowling['wickets'].replace(0, 1)).round(2)

# Filter bowlers with enough matches
bowling = bowling[bowling['matches'] >= 20].reset_index(drop=True)

# Venue effects
df['venue'] = df['venue'].str.strip()

venue_mapping = {
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
}
df['venue'] = df['venue'].replace(venue_mapping)

venue = df.groupby('venue').agg(
    matches       = ('match_id', 'nunique'),
    avg_runs      = ('total_runs', 'mean'),
    total_sixes   = ('batsman_runs', lambda x: (x == 6).sum()),
    total_fours   = ('batsman_runs', lambda x: (x == 4).sum()),
    total_wickets = ('is_wicket', 'sum'),
).reset_index()

venue['runs_per_match']    = (venue['avg_runs'] * 120).round(1)
venue['sixes_per_match']   = (venue['total_sixes'] / venue['matches']).round(2)
venue['wickets_per_match'] = (venue['total_wickets'] / venue['matches']).round(2)

venue = venue[venue['matches'] >= 10].sort_values('runs_per_match', ascending=False)

print("Bowling stats shape:", bowling.shape)
print(bowling[['bowler','wickets','economy','average']].sort_values('wickets', ascending=False).head(10).to_string())
print("\nVenue stats shape:", venue.shape)
print(venue[['venue','matches','runs_per_match','sixes_per_match']].head(5).to_string())


# %% Visualization 1 — Top 5 Batsmen Grouped Bar Comparison
top_batsmen = batting.nlargest(5, 'total_runs')

metrics = ['strike_rate', 'average', 'boundary_pct']
labels  = ['Strike Rate', 'Average', 'Boundary %']
colors  = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

x = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))

for i, (_, row) in enumerate(top_batsmen.iterrows()):
    values = [row['strike_rate'], row['average'], row['boundary_pct']]
    bars = ax.bar(x + i * width, values, width, label=row['batter'],
                  color=colors[i], alpha=0.85, edgecolor='white')

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=7.5, color='dimgray')

ax.set_xticks(x + width * 2)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Top 5 Batsmen — Strike Rate vs Average vs Boundary %',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, max(top_batsmen['strike_rate'].max(),
                   top_batsmen['average'].max()) * 1.15)

sns.despine()
plt.tight_layout()
plt.savefig('charts/batsmen_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
# %% Visualization 2 — Rolling Average (Form Over Time)
# Pick top 4 batsmen by total runs
top_4 = batting.nlargest(4, 'total_runs')['batter'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for idx, player in enumerate(top_4):
    player_df = df[df['batter'] == player].groupby('match_id')['batsman_runs'].sum().reset_index()
    player_df['rolling_avg'] = player_df['batsman_runs'].rolling(window=10, min_periods=1).mean()

    ax = axes[idx]
    ax.plot(player_df.index, player_df['batsman_runs'], color=colors[idx], alpha=0.3, linewidth=1)
    ax.plot(player_df.index, player_df['rolling_avg'], color=colors[idx], linewidth=2.5)
    ax.fill_between(player_df.index, player_df['rolling_avg'], alpha=0.1, color=colors[idx])

    ax.set_title(player, fontsize=13, fontweight='bold')
    ax.set_xlabel('Innings')
    ax.set_ylabel('Runs')
    ax.axhline(player_df['batsman_runs'].mean(), linestyle='--', color='gray', linewidth=1, alpha=0.7)

plt.suptitle('Batting Form — Rolling 10 Match Average', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/rolling_averages.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Visualization 3 — Venue Heatmap
top_venues = venue.nlargest(10, 'matches')

heatmap_data = top_venues[['venue', 'runs_per_match', 'sixes_per_match', 'wickets_per_match']].copy()
heatmap_data['venue'] = heatmap_data['venue'].str.replace(r'\s+', '\n', regex=True)
heatmap_data = heatmap_data.set_index('venue')
heatmap_data.columns = ['Runs/Match', 'Sixes/Match', 'Wickets/Match']

# Normalize for color scaling
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    heatmap_normalized,
    annot=heatmap_data,
    fmt='.1f',
    cmap='RdYlGn',
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Normalized Score'}
)

ax.set_title('Venue Analysis — Top 10 IPL Grounds', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('charts/venue_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
# %% Visualization 4 — Match Impact Score
batting['impact_score'] = (
    (batting['total_runs'] * 0.4) +
    (batting['strike_rate'] * 0.3) +
    (batting['boundary_pct'] * 0.2) +
    (batting['average'] * 0.1)
)

top_impact = batting.nlargest(15, 'impact_score')

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_impact)))

bars = ax.barh(top_impact['batter'], top_impact['impact_score'], color=colors)

for bar, val in zip(bars, top_impact['impact_score']):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{val:.0f}', va='center', fontsize=9)

ax.set_title('Match Impact Score — Top 15 Batsmen', fontsize=15, fontweight='bold')
ax.set_xlabel('Impact Score')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('charts/impact_scores.png', dpi=150, bbox_inches='tight')
plt.show()
# %% Visualization 5 — Season-wise Batting Heatmap
season_batting = df.groupby(['season', 'batter'])['batsman_runs'].sum().reset_index()

top_batters = batting.nlargest(15, 'total_runs')['batter'].tolist()

pivot = season_batting[season_batting['batter'].isin(top_batters)].pivot_table(
    index='batter',
    columns='season',
    values='batsman_runs',
    aggfunc='sum',
    fill_value=0
)

fig, ax = plt.subplots(figsize=(16, 8))

sns.heatmap(
    pivot,
    cmap='YlOrRd',
    annot=True,
    fmt='d',
    linewidths=0.4,
    linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Runs Scored'}
)

ax.set_title('Season-wise Run Tally — Top 15 IPL Batsmen', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('')
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig('charts/season_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Visualization 6 — Economy Rate vs Wickets (Bowlers)
top_bowlers = bowling.nlargest(30, 'wickets')

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    top_bowlers['economy'],
    top_bowlers['wickets'],
    s=top_bowlers['matches'] * 3,
    c=top_bowlers['average'],
    cmap='coolwarm',
    alpha=0.75,
    edgecolors='white',
    linewidth=0.5
)

for _, row in top_bowlers.iterrows():
    ax.annotate(row['bowler'], (row['economy'], row['wickets']),
                fontsize=7, alpha=0.85,
                xytext=(4, 4), textcoords='offset points')

ax.axvline(top_bowlers['economy'].mean(), linestyle='--', color='gray', alpha=0.5)
ax.axhline(top_bowlers['wickets'].mean(), linestyle='--', color='gray', alpha=0.5)

plt.colorbar(scatter, label='Bowling Average')
ax.set_xlabel('Economy Rate', fontsize=12)
ax.set_ylabel('Total Wickets', fontsize=12)
ax.set_title('Bowler Efficiency — Economy vs Wickets', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/bowler_efficiency.png', dpi=150, bbox_inches='tight')
plt.show()
# %% Key Findings
best_batsman   = batting.loc[batting['impact_score'].idxmax(), 'batter']
best_economy   = bowling.loc[bowling['economy'].idxmin(), 'bowler']
best_venue     = venue.loc[venue['runs_per_match'].idxmax(), 'venue']
most_sixes     = venue.loc[venue['sixes_per_match'].idxmax(), 'venue']

print("======================================")
print("   CRICKET PERFORMANCE INTELLIGENCE   ")
print("======================================")
print(f"Highest Impact Batsman  : {best_batsman}")
print(f"Best Economy Bowler     : {best_economy}")
print(f"Most Batting Friendly   : {best_venue}")
print(f"Most Sixes Hit At       : {most_sixes}")
print("======================================")
print("Charts saved in /charts folder")

