# imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# TOP_FOLDER = "../../clean_data"
TOP_FOLDER = "./clean_data"

def check_crossing(time, date):
	"""
	Checks for blockages at crossings at a given time and date.

	Inputs:
	- time (float): Emergency time.
	- date (str): Emergency date in 'YYYY-MM-DD' format.

	Returns:
	- crossings (matrix): A boolean matrix indicating blockages at crossings 
	for 2-minute intervals starting from the input time.
	"""

	# Used to be 10. Each time interval is 2-minutes long.
	NUM_TIME_INTERVS = 10

	blockages = pd.read_csv(TOP_FOLDER + '/merged_blockage_data.csv')
    
	# Convert datetime columns to pandas datetime objects
	blockages['UseBlockageStart'] = pd.to_datetime(blockages['UseBlockageStart'])
	blockages['UseBlockageEnd'] = pd.to_datetime(blockages['UseBlockageEnd'])
	unique_crossings = blockages['CrossingID'].unique()

	# grab the emergency time (start for the crossing) and 2 min intervals
	start_time = pd.to_datetime(f"{date} {time}")
	time_intervals = [start_time + timedelta(minutes=2 * i) for i in range(NUM_TIME_INTERVS)]
	formatted_time_intervals = [t.strftime('%Y-%m-%d %H:%M') for t in time_intervals]

	# go through each time intervals (forward and backwards)
	crossings = pd.DataFrame(0, index=unique_crossings, columns=formatted_time_intervals)
	for i, (t_start, t_end) in enumerate(zip(time_intervals[:-1], time_intervals[1:])):
		
		# Filter blockages that overlap with the current interval
		mask = (
			(blockages['UseBlockageStart'] < t_end) &
			(blockages['UseBlockageEnd'] > t_start) &
			(blockages['UseBlockageStart'].dt.date == start_time.date())
		)

		# Calculate how long the blockage lasts
		# blockages['BlockageClear'] - blockages['BlockageStart']
		
		# Get the relevant rows
		relevant_blockages = blockages[mask]
		for _, row in relevant_blockages.iterrows():
			crossing_id = row['CrossingID']
			# Calculate how long the blockage lasts
			cur_blockage_time = row['UseBlockageEnd'] - row['UseBlockageStart']
			cur_blockage_time = cur_blockage_time.total_seconds() / 60.0

			crossings.loc[crossing_id, formatted_time_intervals[i]] = 1 # mark crossing as blocked
			# crossings.loc[crossing_id, formatted_time_intervals[i]] = cur_blockage_time # mark crossing as blocked

	crossings = crossings.reset_index().rename(columns={'index': 'CrossingID'})

	return crossings



if __name__ == "__main__":
	# run tests for check_crossing()
	test = check_crossing("18:30", "2019-01-17")
	# print(test)
	# print(type(test))
	