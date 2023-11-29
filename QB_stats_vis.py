# Import scraping modules
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Import data manipulation modules
import pandas as pd
import numpy as np

# Import data visualization modules
import matplotlib as mpl
import matplotlib.pyplot as plt

# URL of page
url_2019 = 'https://www.pro-football-reference.com/years/2019/passing.htm'
url_2020 = 'https://www.pro-football-reference.com/years/2020/passing.htm'

# Open URL and pass to BeautifulSoup
html_2019 = urlopen(url_2019)
stats_page_2019 = BeautifulSoup(html_2019)
html_2020 = urlopen(url_2020)
stats_page_2020 = BeautifulSoup(html_2020)

# Collect table headers
column_headers_2019 = stats_page_2019.findAll('tr')[0]
column_headers_2019= [i.getText() for i in column_headers_2019.findAll('th')]
column_headers_2020 = stats_page_2020.findAll('tr')[0]
column_headers_2020 = [i.getText() for i in column_headers_2020.findAll('th')]

# Collect table rows
rows_2019 = stats_page_2019.findAll('tr')[1:]
rows_2020 = stats_page_2020.findAll('tr')[1:]

# Get stats from each row
qb_stats_2019 = []
for i in range(len(rows_2019)):
    qb_stats_2019.append([col.getText() for col in rows_2019[i].findAll('td')])
qb_stats_2020 = []
for i in range(len(rows_2020)):
    qb_stats_2020.append([col.getText() for col in rows_2020[i].findAll('td')])

# Create DataFrame from our scraped data
data_all_2019 = pd.DataFrame(qb_stats_2019, columns=column_headers_2019[1:])
data_all_2020 = pd.DataFrame(qb_stats_2020, columns=column_headers_2020[1:])

# Examine first five rows of data
data_all_2019.head()
data_all_2020.head()

# View columns in data
data_all_2019.columns
data_all_2020.columns

# Rename sack yards column to `Yds_Sack`
new_columns_2019 = data_all_2019.columns.values
new_columns_2019[-6] = 'Yds_Sack'
data_all_2019.columns = new_columns_2019
new_columns_2020 = data_all_2020.columns.values
new_columns_2020[-6] = 'Yds_Sack'
data_all_2020.columns = new_columns_2020

# Filter by number of games played (for 2019) and by greater than 1500 yards (2020)
data_2019 = data_all_2019[pd.to_numeric(data_all_2019['Yds']) > 1500]
data_2020 = data_all_2020[pd.to_numeric(data_all_2020['G']) > 2]

# View columns in data
data_2019.columns
data_2020.columns

# Select stat categories
categories = ['Cmp%', 'Yds', 'TD', 'Int', 'Y/A', 'Rate']

# Create data subset for radar chart
data_radar_2019 = data_2019[['Player', 'Tm'] + categories]
data_radar_2019.head()
data_radar_2020 = data_2020[['Player', 'Tm'] + categories]
data_radar_2020.head()

# Check data types
data_radar_2019.dtypes
data_radar_2020.dtypes

# Convert data to numerical values
for i in categories:
    data_radar_2019[i] = pd.to_numeric(data_radar_2019[i])
    data_radar_2020[i] = pd.to_numeric(data_radar_2020[i])

# Check data types
data_radar_2019.dtypes
data_radar_2020.dtypes

# Remove ornamental characters for achievements
data_radar_2019['Player'] = data_radar_2019['Player'].str.replace('*', '')
data_radar_2019['Player'] = data_radar_2019['Player'].str.replace('+', '')
data_radar_2019.head(32)
data_radar_2020['Player'] = data_radar_2020['Player'].str.replace('*', '')
data_radar_2020['Player'] = data_radar_2020['Player'].str.replace('+', '')
data_radar_2020.head(32)

# Create columns with percentile rank
for i in categories:
    data_radar_2019[i + '_Rank'] = data_radar_2019[i].rank(pct=True)
    data_radar_2020[i + '_Rank'] = data_radar_2020[i].rank(pct=True)

# We need to flip the rank for interceptions
data_radar_2019['Int_Rank'] = 1 - data_radar_2019['Int_Rank']
data_radar_2020['Int_Rank'] = 1 - data_radar_2020['Int_Rank']

# Examine data
data_radar_2019.head()
data_radar_2020.head()

# General plot parameters
mpl.rcParams['font.family'] = 'Avenir'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.linewidth'] = 0
mpl.rcParams['xtick.major.pad'] = 15

team_colors = {'ARI':'#97233f', 'ATL':'#a71930', 'BAL':'#241773', 'BUF':'#00338d',
               'CAR':'#0085ca', 'CHI':'#0b162a', 'CIN':'#fb4f14', 'CLE':'#311d00',
               'DAL':'#041e42', 'DEN':'#002244', 'DET':'#0076b6', 'GNB':'#203731',
               'HOU':'#03202f', 'IND':'#002c5f', 'JAX':'#006778', 'KAN':'#e31837',
               'LAC':'#002a5e', 'LAR':'#003594', 'MIA':'#008e97', 'MIN':'#4f2683',
               'NWE':'#002244', 'NOR':'#d3bc8d', 'NYG':'#0b2265', 'NYJ':'#125740',
               'OAK':'#000000', 'PHI':'#004c54', 'PIT':'#ffb612', 'SFO':'#aa0000',
               'SEA':'#002244', 'TAM':'#d50a0a', 'TEN':'#0c2340', 'WAS':'#773141'}



# Calculate angles for radar chart
offset = np.pi/6
angles = np.linspace(0, 2*np.pi, len(categories) + 1) + offset

# Function to create radar chart


def create_radar_chart(ax, angles, player_data, color='blue'):
    # Plot data and fill with team color
    ax.plot(angles, np.append(player_data[-(len(angles) - 1):], player_data[-(len(angles) - 1)]), color=color,
            linewidth=2)
    ax.fill(angles, np.append(player_data[-(len(angles) - 1):], player_data[-(len(angles) - 1)]), color=color,
            alpha=0.2)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Remove radial labels
    ax.set_yticklabels([])

    # Add player name
    ax.text(np.pi / 2, 1.7, player_data[0], ha='center', va='center', size=18, color=color)

    # Use white grid
    ax.grid(color='white', linewidth=1.5)

    # Set axis limits
    ax.set(xlim=(0, 2 * np.pi), ylim=(0, 1))

    return ax

# Function to get QB data
def get_qb_data(data, team):
    return np.asarray(data[data['Tm'] == team])[0]

# NFC West 2019
# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='white')

# Add subplots
ax1 = fig.add_subplot(221, projection='polar', facecolor='#ededed')
ax2 = fig.add_subplot(222, projection='polar', facecolor='#ededed')
ax3 = fig.add_subplot(223, projection='polar', facecolor='#ededed')
ax4 = fig.add_subplot(224, projection='polar', facecolor='#ededed')

# Adjust space between subplots
plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Get QB data
sf_data = get_qb_data(data_radar_2019, 'SFO')
sea_data = get_qb_data(data_radar_2019, 'SEA')
ari_data = get_qb_data(data_radar_2019, 'ARI')
lar_data = get_qb_data(data_radar_2019, 'LAR')

# Plot QB data
ax1 = create_radar_chart(ax1, angles, lar_data, team_colors['LAR'])
ax2 = create_radar_chart(ax2, angles, ari_data, team_colors['ARI'])
ax3 = create_radar_chart(ax3, angles, sea_data, team_colors['SEA'])
ax4 = create_radar_chart(ax4, angles, sf_data, team_colors['SFO'])
#plt.title('NFC West 2019')
plt.show()



# AFC East 2019
# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='white')

# Add subplots
ax1 = fig.add_subplot(221, projection='polar', facecolor='#ededed')
ax2 = fig.add_subplot(222, projection='polar', facecolor='#ededed')
ax3 = fig.add_subplot(223, projection='polar', facecolor='#ededed')
ax4 = fig.add_subplot(224, projection='polar', facecolor='#ededed')

# Adjust space between subplots
plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Get QB data
nwe_data = get_qb_data(data_radar_2019, 'NWE')
mia_data = get_qb_data(data_radar_2019, 'MIA')
nyj_data = get_qb_data(data_radar_2019, 'NYJ')
buf_data = get_qb_data(data_radar_2019, 'BUF')

# Plot QB data
ax1 = create_radar_chart(ax1, angles, nwe_data, team_colors['NWE'])
ax2 = create_radar_chart(ax2, angles, mia_data, team_colors['MIA'])
ax3 = create_radar_chart(ax3, angles, nyj_data, team_colors['NYJ'])
ax4 = create_radar_chart(ax4, angles, buf_data, team_colors['BUF'])
#plt.title('AFC East 2019')
plt.show()







# NFC West 2020
# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='white')

# Add subplots
ax1 = fig.add_subplot(221, projection='polar', facecolor='#ededed')
ax2 = fig.add_subplot(222, projection='polar', facecolor='#ededed')
ax3 = fig.add_subplot(223, projection='polar', facecolor='#ededed')
ax4 = fig.add_subplot(224, projection='polar', facecolor='#ededed')

# Adjust space between subplots
plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Get QB data
sf_data = get_qb_data(data_radar_2020, 'SFO')
sea_data = get_qb_data(data_radar_2020, 'SEA')
ari_data = get_qb_data(data_radar_2020, 'ARI')
lar_data = get_qb_data(data_radar_2020, 'LAR')

# Plot QB data
ax1 = create_radar_chart(ax1, angles, lar_data, team_colors['LAR'])
ax2 = create_radar_chart(ax2, angles, ari_data, team_colors['ARI'])
ax3 = create_radar_chart(ax3, angles, sea_data, team_colors['SEA'])
ax4 = create_radar_chart(ax4, angles, sf_data, team_colors['SFO'])
#plt.title('NFC West 2020')
plt.show()






# AFC East 2020
# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='white')

# Add subplots
ax1 = fig.add_subplot(221, projection='polar', facecolor='#ededed')
ax2 = fig.add_subplot(222, projection='polar', facecolor='#ededed')
ax3 = fig.add_subplot(223, projection='polar', facecolor='#ededed')
ax4 = fig.add_subplot(224, projection='polar', facecolor='#ededed')

# Adjust space between subplots
plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Get QB data
nwe_data = get_qb_data(data_radar_2020, 'NWE')
mia_data = get_qb_data(data_radar_2020, 'MIA')
nyj_data = get_qb_data(data_radar_2020, 'NYJ')
buf_data = get_qb_data(data_radar_2020, 'BUF')

# Plot QB data
ax1 = create_radar_chart(ax1, angles, nwe_data, team_colors['NWE'])
ax2 = create_radar_chart(ax2, angles, mia_data, team_colors['MIA'])
ax3 = create_radar_chart(ax3, angles, nyj_data, team_colors['NYJ'])
ax4 = create_radar_chart(ax4, angles, buf_data, team_colors['BUF'])
#plt.title('AFC East 2020')
plt.show()

print('done')

