import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load in our raw data
df1 = pd.read_csv(r'./raw_data/TRAINFO/HoustonBlockageData.csv')
df2 = pd.read_csv(r'./raw_data/TRAINFO/HoustonBlockageData2.csv')
blockage_data = pd.concat([df1, df2], ignore_index=True)

# Convert the CrossingID and Street to string
blockage_data['CrossingID'] = blockage_data['CrossingID'].astype(str)
blockage_data['Street'] = blockage_data['Street'].astype(str)

# Convert the BlockageStart and BlockageClear to datetime
blockage_data["BlockageClear"] = pd.to_datetime(blockage_data["BlockageClear"], format = '%Y-%m-%d %H:%M:%S')
blockage_data['BlockageStart'] = pd.to_datetime(blockage_data['BlockageStart'], format = '%Y-%m-%d %H:%M:%S')
blockage_data['BlockageDuration'] = (blockage_data['BlockageClear'] - blockage_data['BlockageStart']).dt.total_seconds()
map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
blockage_data['DoW'] = blockage_data['BlockageStart'].dt.dayofweek.map(map)

CrossingID_list = blockage_data['CrossingID'].unique()

def visualize_mean_blockage_time():
    """
    Creates a plot with:
    CrossingID in the x-axis and mean blockage time in the y-axis (with variance visualized).

    Inputs: None
    
    Returns none.
    """
    # get mean and variance
    ind_mean = blockage_data.groupby('CrossingID')['BlockageDuration'].mean()
    ind_var = blockage_data.groupby('CrossingID')['BlockageDuration'].var()
    mean = blockage_data['BlockageDuration'].mean()

    # start plotting
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    bars = ax1.bar(range(len(ind_mean)), ind_mean, color='g', alpha=0.4)
    ax1.axhline(mean, color='r', linestyle='--', label=f'Overall blockage time Mean {mean}')
    line = ax2.plot(range(len(ind_var)), ind_var, color='b', label='Blockage time Variance')
    plt.xticks(range(len(ind_mean)), rotation=90)
    ax1.set_xlabel('Crossing ID')
    ax1.set_ylabel('Mean Blockage Duration')
    ax2.set_ylabel('Blockage Duration Variance')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    plt.title("Average Blockage Times for all Crossings")
    plt.show()
    print(f'The top 5 crossings with the longest blockage time are \n {ind_mean.nlargest(5)}')

#visualize_mean_blockage_time()


def visualize_blalock_blockage_duration_intervals():
    """
    Constructs a bar graph of Blalock Road's blockages.
    Each bin represents the duration of the blockage (i.e 0-5 mins; 5-10 mins; etc)
    and its height is how many total blockages there have been of that duration.

    Inputs: None

    Returns none.
    """
    # Blalock Road basic statistics
    #print(f'The Street with Crossing ID 743658M is: {blockage_data[blockage_data["CrossingID"] == "743658M"]["Street"].unique()}')

    crossing_data = blockage_data[blockage_data['CrossingID'] == '743658M']
    #print(f'The basic statistics of the blockage time for crossing 743658M are: \n {crossing_data["BlockageDuration"].describe()}')

    # create bins and labels
    bins = [0, 300, 600, 1200, 1800, 2400, 3600, float('inf')]  
    labels = ['0-5min', '5-10min', '10-20min', '20-30min', '30-40min', '40-60min', '60min+']

    crossing_data_binned = pd.cut(crossing_data['BlockageDuration'], bins=bins, labels=labels)

    # get the basic interval stats
    interval_stats = pd.DataFrame({
        'Count': crossing_data_binned.value_counts().sort_index(),
        'Percentage': crossing_data_binned.value_counts(normalize=True).sort_index() * 100
    })

    plt.figure(figsize=(12, 6))
    interval_stats['Count'].plot(kind='bar')
    plt.title('Blockage Duration Intervals for Blalock Road (CrossingID 743658M)')
    plt.xlabel('Duration Intervals')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_blalock_blockage_duration_intervals_truncated():
    """
    Constructs a bar graph of Blalock Road's blockages, truncated at 30 minutes.
    Each bin represents a 1-minute duration (e.g., 0-1 min, 1-2 min, etc), and
    the height is how many total blockages have been recorded in that duration range.

    Inputs: None
    Returns: None
    """
    # Filter for the relevant crossing
    crossing_data = blockage_data[blockage_data['CrossingID'] == '743658M']

    # Only keep blockages that lasted 30 minutes or less (1800 seconds)
    crossing_data = crossing_data[crossing_data['BlockageDuration'] <= 1800]

    # Create 1-minute bins (0 to 1800 seconds, step = 60)
    bins = list(range(0, 1860, 60))  # 1860 to include up to 1800
    labels = [f'{i}-{i+1} min' for i in range(0, 30)]

    # Bin the data
    crossing_data_binned = pd.cut(crossing_data['BlockageDuration'], bins=bins, labels=labels, right=False)

    # Get stats per interval
    interval_stats = pd.DataFrame({
        'Count': crossing_data_binned.value_counts().sort_index(),
        'Percentage': crossing_data_binned.value_counts(normalize=True).sort_index() * 100
    })

    # Plot the results
    plt.figure(figsize=(14, 6))
    interval_stats['Count'].plot(kind='bar')
    plt.title('Blockage Duration Intervals for Blalock Road (<= 30 Minutes, 1-Minute Chunks)')
    plt.xlabel('Duration Intervals')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# visualize_blalock_blockage_duration_intervals_truncated()

def get_5sec_empirical_distribution_for_crossing(crossing_id):
    """
    Returns the empirical probability distribution of blockage durations for a specific crossing
    using 5-second bins, truncated at 30 minutes (1800 seconds).

    Parameters:
    - crossing_id (str): The ID of the crossing to extract data for.

    Returns:
    - probabilities (np.array): Normalized probability array over 5-sec bins up to 30 minutes.
    - bin_edges (list of int): List of bin edge start times (in seconds).
    """
    # Filter and truncate to ≤ 30 min
    crossing_data = blockage_data[
        (blockage_data['CrossingID'] == crossing_id) &
        (blockage_data['BlockageDuration'] <= 1800)
    ].copy()

    # Bin into 5-second intervals
    bin_edges = list(range(0, 1805, 5))  # 0s to 1800s inclusive
    crossing_data['Bin'] = pd.cut(
        crossing_data['BlockageDuration'],
        bins=bin_edges,
        right=False,
        include_lowest=True
    )

    # Count and normalize
    counts = crossing_data['Bin'].value_counts().sort_index()
    probabilities = counts.values / counts.sum()

    return probabilities, bin_edges


def visualize_blockage_duration_mean_and_variance():
    """
    Visualizes the blockage time mean for each day in a bar chart overlayed
    with a variance detailing the variance.

    Input:
    - None

    Returns none.
    """

    # grab the blockage mean and variance
    mean = blockage_data['BlockageDuration'].mean()
    ind_mean = blockage_data.groupby('DoW')['BlockageDuration'].mean()
    ind_var = blockage_data.groupby('DoW')['BlockageDuration'].var()
    DoW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # start plotting -- get axes
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    bars = ax1.bar(range(len(ind_mean)), ind_mean, color='g', alpha=0.4)
    ax1.axhline(mean, color='r', linestyle='--', label=f'Overall blockage time Mean {mean}')
    line = ax2.plot(range(len(ind_var)), ind_var, color='b', label='Blockage time Variance')
    plt.xticks(range(len(ind_mean)), DoW, rotation=90)

    # set labels, create legend, and display plot
    ax1.set_xlabel('Day of Week')
    ax2.set_ylabel('Mean Blockage Duration')
    ax2.set_ylabel('Blockage Duration Variance')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()

def visualize_average_hourly_blockage_duration():
    """
    Visualizes the average blockage duration per hour of the day.

    Inputs:
    - None

    Returns none.
    """
    # peak hour analysis
    blockage_data['Hour'] = blockage_data['BlockageStart'].dt.hour
    blockage_data['Month'] = blockage_data['BlockageStart'].dt.month
    blockage_data['Day'] = blockage_data['BlockageStart'].dt.day

    # Map days of the week
    blockage_data['DoW'] = blockage_data['BlockageStart'].dt.dayofweek.map(map)
    filtered_data = blockage_data[blockage_data['BlockageDuration'] <= 3600]
    hourly_avg = blockage_data.groupby('Hour')['BlockageDuration'].mean()
    filtered_avg = filtered_data.groupby('Hour')['BlockageDuration'].mean()

    # plot avg blockage duration by hour with (and outliers taken out)
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    hours = np.arange(24)

    # create our two overlayed bars
    plt.bar(hours - bar_width/2, hourly_avg, width=bar_width, color='skyblue', alpha=0.7, label="All Data")
    plt.bar(hours + bar_width/2, filtered_avg, width=bar_width, color='orange', alpha=0.7, label="Filtered Data (≤ 1 Hour)")

    # create labels, title, and display plot
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Blockage Duration (Seconds)")
    plt.title("Average Train Blockage Duration by Hour (Comparison)")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def visualize_blockage_frequency_hourly_for_crossing(crossing_id):
    """
    Visualizes the blockage frequency distribution for a specific crossing over hours of the day.
    
    Inputs:
    - crossing_id (str): The ID of the crossing to visualize.

    Returns none.
    """
    crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id]
    crossing_data = crossing_data.set_index('BlockageStart')
    crossing_data = crossing_data.resample('H').count()
    crossing_data = crossing_data[['CrossingID']]
    crossing_data.columns = ['BlockageCount']
    
    crossing_data.plot()
    plt.title(f'Blockage Frequency for Crossing {crossing_id}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Blockage Frequency')
    plt.show()

def visualize_average_monthly_blockage_duration():
    """
    Plots the average train blockage duration per month, compares it to the overall average,
    and prints related statistics including per-month variance and the longest blockages in May.

    Inputs:
    - None

    Returns none.
    """
    # to see how much we are samples we are averaging from
    #monthly_counts = blockage_data.groupby('Month')['BlockageDuration'].count()
    #print(f"Number of blockage entries per month:\n{monthly_counts}")

    # get mean and var
    monthly_avg = blockage_data.groupby('Month')['BlockageDuration'].mean()
    overall_avg = monthly_avg.mean()
    monthly_variance = blockage_data.groupby('Month')['BlockageDuration'].var()
    print(f"Variance of blockage duration per month (in seconds): {monthly_variance.to_list()}")

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_avg.index, monthly_avg.values, color='b', alpha=0.7)
    plt.axhline(y=overall_avg, color='red', linestyle='dotted', linewidth=2, label=f'Overall Avg: {overall_avg:.2f} sec')
    plt.xlabel("Month")
    plt.ylabel("Average Blockage Duration (Seconds)")
    plt.title("Average Train Blockage Duration by Month")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.xticks(ticks=range(1, 13), labels=month_labels, rotation=45)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # get a peek into may data since it is so much higher
    may_data = blockage_data[blockage_data['Month'] == 5]
    #top_50_may = may_data.nlargest(50, 'BlockageDuration')['BlockageDuration']
    #print(f"Top 50 highest blockage times in May: {top_50_may.to_list()}")


def visualize_blockage_duration_for_crossing(crossing_id):
    """
    Visualizes the blockage duration distribution for a specific crossing using seaborn.
    
    Inputs:
    - blockage_data (DataFrame): The dataset containing blockage records.
    - crossing_id (str): The ID of the crossing to visualize.

    Returns none.
    """
    crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id]
    
    sns.histplot(crossing_data['BlockageDuration'], bins=50)
    plt.title(f'Blockage Duration for Crossing {crossing_id}')
    plt.xlabel('Blockage Duration')
    plt.ylabel('Frequency')
    plt.show()

# visualize_blockage_duration_for_crossing('755709E') -- input test that Bayzhan wrote

def visualize_blockage_frequency_per_day_for_crossing(crossing_id):
    """
    Visualizes the blockage frequency distribution for a specific crossing, aggregated per day.

    Inputs:
    - crossing_id (str): The ID of the crossing to visualize.

    Returns none.
    """
    # Filter data for the specific crossing
    crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id].copy()

    # Debugging: Print if no data is found
    if crossing_data.empty:
        print(f"No data found for Crossing ID: {crossing_id}")
        return

    # Ensure 'BlockageStart' is a datetime column
    crossing_data['BlockageStart'] = pd.to_datetime(crossing_data['BlockageStart'], errors='coerce')

    # Drop NaT values after conversion (if any)
    crossing_data = crossing_data.dropna(subset=['BlockageStart'])

    # Extract only the date (removes time part)
    crossing_data['Date'] = crossing_data['BlockageStart'].dt.date

    # Aggregate data per day (count occurrences)
    blockage_counts = crossing_data.groupby('Date').size().reset_index(name='BlockageCount')

    # Debugging: Print if no blockage events are found
    if blockage_counts.empty:
        print(f"No blockage events recorded for Crossing ID: {crossing_id}")
        return

    # Convert Date back to datetime for better plotting
    blockage_counts['Date'] = pd.to_datetime(blockage_counts['Date'])

    # Create Seaborn bar plot
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=blockage_counts, x="Date", y="BlockageCount")

    # Format x-axis labels
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Blockage Frequency")
    plt.title(f"Daily Blockage Frequency for Crossing {crossing_id}")

    plt.show()

def visualize_blockage_distribution():
    """
    Visualizes the blockage distribution for all crossings, aggregated by duration.

    Inputs:
    - None

    Returns none.
    """
    ### plotting bar graph with bins representing blockage times (secs)
    bins = [0, 300, 600, 1800, 3600, float('inf')]
    labels = ['0-5 min', '5-10 min', '10-30 min', '30-60 min', '60+ min']

    # getting data and plotting
    blockage_data['DurationCategory'] = pd.cut(blockage_data['BlockageDuration'], bins=bins, labels=labels, right=False)
    duration_counts = blockage_data['DurationCategory'].value_counts().reindex(labels, fill_value=0)

    # get the actual count on the graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(duration_counts.index, duration_counts.values, color='blue', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel("Blockage Duration Interval")
    plt.ylabel("Number of Blockages")
    plt.title("Distribution of Train Blockages by Duration Interval")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def visualize_five_minute_intervals_for_crossing(crossing_id):
    """
    Visualizes the number of recorded blockage events for each 5-minute interval of the day
    using Seaborn, with improved x-axis readability and a color gradient from the "crest" palette.

    Inputs:
    - crossing_id (str): The ID of the crossing to visualize.

    Returns none.
    """
    # Filter data for the specific crossing
    crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id]

    # Count occurrences of each 5-minute interval
    interval_counts = crossing_data['FiveMinInterval'].value_counts().sort_index()

    # Convert interval index to readable time format
    interval_labels = [(f"{(i // 12):02d}:{(i % 12) * 5:02d}") for i in interval_counts.index]

    # Create a color gradient from the "crest" palette
    # colors = sns.color_palette("crest", as_cmap=True)(np.linspace(0, 1, len(interval_counts)))
    
    # Create Seaborn bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=interval_labels, y=interval_counts.values, color = "skyblue")
    
    # Improve x-axis readability by reducing labels
    step = max(1, len(interval_labels) // 20)  # Show only 20 labels max
    plt.xticks(np.arange(0, len(interval_labels), step), interval_labels[::step], rotation=90)

    plt.xlabel("Time of Day (5-minute intervals)")
    plt.ylabel("Recorded Blockage Events")
    plt.title(f"Blockage Event Frequency for Crossing {crossing_id}")

    plt.show()

def visualize_weekday_frequency():
    """
    Visualizes the number of recorded blockage events per weekday for all stations
    using Seaborn.

    Inputs:
    - blockage_data (DataFrame): The dataset containing blockage records.

    Returns none.
    """
    # Convert timestamp to weekday
    blockage_data['Weekday'] = blockage_data['BlockageStart'].dt.day_name()
    
    # Count occurrences per weekday
    weekday_counts = blockage_data['Weekday'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    
    # Create Seaborn bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, color = "skyblue")
    
    plt.xlabel("Weekday")
    plt.ylabel("Recorded Blockage Events")
    plt.title("Blockage Event Frequency per Weekday for All Stations")
    
    plt.show()

def visualize_blockage_distribution_months_for_crossing(crossing_id):
    """
    Visualizes the blockage time distribution for a specific crossing over days and months.
    
    Inputs:
    - crossing_id (str): The ID of the crossing to visualize.

    Returns none.
    """
    crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id]
    crossing_data = crossing_data.set_index('BlockageStart')
    crossing_data = crossing_data.resample('D').count()
    crossing_data = crossing_data[['CrossingID']]
    crossing_data.columns = ['BlockageCount']
    
    crossing_data.plot()
    plt.title(f'Blockage Count for Crossing {crossing_id}')
    plt.xlabel('Date')
    plt.ylabel('Blockage Count')
    plt.show()

# visualize_blockage_distribution_months_for_crossing('755709E') -- input test that Bayzhan wrote

# running this condition so we can run this file individually -- these print statements won't be run
# when we import this module elsewhere (like in our demo notebook)
if __name__ == "__main__":
    print(f'There are {blockage_data["CrossingID"].nunique()} unique crossings in the data')
    print(f'Blockage Raw Data has {blockage_data.shape[0]} records')
    print(f'The Feature in Blockage Raw Data are: \n {blockage_data.columns}')

    # FIND THE TOP THREE CROSSINGS AND DRAW MORE INFO ABOUT THEM
    top_crossings = ['743658M', '859568M', '758739G']
    crossing_lookup = blockage_data.set_index('CrossingID')['Street'].to_dict()

    # get the street names given the top crossing IDs and the top 3 times
    for crossing_id in top_crossings:
        print(f"{crossing_id}: {crossing_lookup.get(crossing_id, 'Not Found')}")

        # get the top 5 highest times and print them out
        crossing_data = blockage_data[blockage_data['CrossingID'] == crossing_id]
        top_5_blockages = crossing_data.nlargest(5, 'BlockageDuration')['BlockageDuration'] / 3600
        print(f"    Top 5 blockage times (in hours): {top_5_blockages.to_list()}\n")

    # verify blalock road top data... 365 is A LOT
    blalock_data = blockage_data[blockage_data['Street'].str.contains('BLALOCK ROAD', case=False, na=False)]
    top_blalock_time = blalock_data['BlockageDuration'].max()
    print("Blalock Road Statistics:")
    print(f"Top blockage duration for Blalock Road: {top_blalock_time} seconds!!!")
    print(f"This is {top_blalock_time / 3600:.2f} hours ... which is {top_blalock_time / 86400:.2f} days!!!\n")

    # get month day year of this top time
    top_blalock_row = blalock_data.loc[blalock_data['BlockageDuration'].idxmax()]
    blockage_start = top_blalock_row['BlockageStart']
    blockage_end = top_blalock_row['BlockageClear']
    blockage_duration_hours = top_blalock_row['BlockageDuration'] / 3600
    print(f"Top Blockage on Blalock Road:")
    print(f"   Start Time: {blockage_start} (Year: {blockage_start.year}, Month: {blockage_start.month}, Day: {blockage_start.day}, Time: {blockage_start.time()})")
    print(f"   End Time: {blockage_end} (Year: {blockage_end.year}, Month: {blockage_end.month}, Day: {blockage_end.day}, Time: {blockage_end.time()})")
    print(f"   Duration: {blockage_duration_hours:.2f} hours ({blockage_duration_hours / 24:.2f} days)")


    # variance of blalock road must be insanely high because of this 
    blalock_std = blalock_data['BlockageDuration'].std()
    blalock_variance = blalock_data['BlockageDuration'].var()
    print(f"Blockage Duration Statistics for Blalock Road:")
    print(f"   Standard Deviation: {blalock_std:.2f} seconds ({blalock_std / 3600:.2f} hours)")
    print(f"   Variance: {blalock_variance:.2f} seconds² ({blalock_variance / 3600:.2f} hours²)")

