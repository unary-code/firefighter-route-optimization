import pandas as pd
import numpy as np
import os


def preprocess_blockage_data(input_csv, output_csv):
    """
    Reads the TRAINFO raw blockage data and adds basic additional features (columns) described below.
    - BlockageDuration (seconds), Day of Week (Mon, Tue, . . .), Hour, Month, Day

    Inputs:
    - input_csv (str): Path to the raw blockage data CSV.
    - output_csv (str): Path where the cleaned CSV will be saved.

    Returns:
    - blockage_data_cleaned (pandas df): Cleaned dataset.
    """
    try:
        blockage_data = pd.read_csv(input_csv, parse_dates=['BlockageStart', 'BlockageClear'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The input CSV file '{input_csv}' was not found.")
    
    blockage_data['BlockageDuration'] = (blockage_data['BlockageClear'] - blockage_data['BlockageStart']).dt.total_seconds()

    # Get dates for new columns
    blockage_data['DoW'] = blockage_data['BlockageStart'].dt.strftime('%a') 
    blockage_data['Hour'] = blockage_data['BlockageStart'].dt.hour
    blockage_data['Month'] = blockage_data['BlockageStart'].dt.month
    blockage_data['Day'] = blockage_data['BlockageStart'].dt.day

    # Export the cleaned dataset
    try:
        blockage_data.to_csv(output_csv, index=False)
    except:
        raise IOError(f"Tried to export cleaned blockage data to {output_csv} but an error occured.")
    return blockage_data


def transform_blockage_matrix(blockage_data, output_csv, start_time, end_time, target_day):
    """
    Transforms blockage data into a matrix that will be used as an input for
    our prediction model:
    - Rows: crossing IDs
    - Columns: 5-minute time intervals (e.g., "10:00-10:05", "10:05-10:10")
    - Values: Total blockage duration per interval (in seconds)

    Inputs:
    - blockage_data (pandas df): Cleaned blockage dataset.
    - output_csv (str): Path where the transformed matrix CSV will be saved.
    - start_time (str): Start time for filtering (e.g., "10:00").
    - end_time (str): End time for filtering (e.g., "15:00").
    - target_day (str): Specific date to analyze (e.g., "2025-01-17").

    Returns: 
    - blockage_matrix (List[[]]): the transformed blockage matrix 
    - blockage_numpy_matrix (NumPy array): NumPy version of the matrix
    """

    # filter for a specific day and within the given interval
    blockage_data = blockage_data[blockage_data["BlockageStart"].dt.strftime('%Y-%m-%d') == target_day].copy()
    blockage_data_filtered = blockage_data[(blockage_data["BlockageStart"] < end_time) &
        (blockage_data["BlockageClear"] > start_time)]
    #print(f"Blockages within {start_time}-{end_time}: {blockage_data_filtered.shape}")

    # TODO: error check if there is enough data for this particular day
    # get the time windows as our columns
    time_slots = pd.date_range(start=start_time, end=end_time, freq="5T")
    time_intervals = [f"{t.strftime('%H:%M')}-{(t + pd.Timedelta(minutes=5)).strftime('%H:%M')}" for t in time_slots[:-1]]
    unique_crossings = blockage_data["CrossingID"].unique()
    # print(f"Generated time intervals: {time_intervals[:5]} ...")

    # start building our solution matrix 
    blockage_matrix = pd.DataFrame(0, index=unique_crossings, columns=time_intervals)

    for _, row in blockage_data.iterrows():
        crossing_id = row["CrossingID"]
        start, end = row["BlockageStart"], row["BlockageClear"]
        # print(f"Currently at: CrossingID: {crossing_id}, Blockage Start: {start}, Blockage End: {end}")

        # now we can distribute the blockage duration across the intervals
        curr_time = start.floor("5T")
        while curr_time < end:
            next_time = curr_time + pd.Timedelta(minutes=5)

            # find the time spent in the current interval
            duration_seconds = (min(next_time, end) - max(curr_time, start)).total_seconds()
            if duration_seconds > 0:
                time_slot_str = f"{curr_time.strftime('%H:%M')}-{next_time.strftime('%H:%M')}"
                # print(f"Adding {duration_seconds} sec to time slot {time_slot_str}")
                
                # update the time spent in this particular interval
                if time_slot_str in blockage_matrix.columns:
                    blockage_matrix.loc[crossing_id, time_slot_str] += duration_seconds
            curr_time = next_time

    blockage_numpy_matrix = blockage_matrix.to_numpy() # save both numpy and normal version
    blockage_matrix.to_csv(output_csv, index=True)
    # print(f"Transformed blockage matrix for {target_day} saved to {output_csv}")
    return blockage_matrix, blockage_numpy_matrix


def average_blockage_matrix(blockage_data, output_csv, start_time, end_time):
    """
    Transforms blockage data into a matrix that will be used as an input for
    our prediction model:
    - Rows: crossing IDs
    - Columns: 5-minute time intervals (e.g., "10:00-10:05", "10:05-10:10")
    - Values: Average blockage duration per interval (in seconds) across all days

    Inputs:
    - blockage_data (pandas df): Cleaned blockage dataset.
    - output_csv (str): Path where the transformed matrix CSV will be saved.
    - start_time (str): Start time for filtering (e.g., "10:00").
    - end_time (str): End time for filtering (e.g., "15:00").

    Returns:
    - averaged_matrix (List[[]]): accumulation of average blockage times from start_time to end_time.
    - blockage_numpy_matrix (NumPy array): NumPy version of averaged_matrix.
    """

    # Get all unique days and the time window
    blockage_data["Date"] = blockage_data["BlockageStart"].dt.strftime('%Y-%m-%d')
    unique_dates = blockage_data["Date"].unique()
    time_slots = pd.date_range(start=start_time, end=end_time, freq="5T")
    time_intervals = [f"{t.strftime('%H:%M')}-{(t + pd.Timedelta(minutes=5)).strftime('%H:%M')}" for t in time_slots[:-1]]
    unique_crossings = blockage_data["CrossingID"].unique()

    # start building our cumulative matrix
    cumulative_matrix = pd.DataFrame(0.0, index=unique_crossings, columns=time_intervals)
    day_counts = pd.Series(0, index=unique_crossings)

    for day in unique_dates:
        day_data = blockage_data[blockage_data["Date"] == day].copy()
        if day_data.empty: # might need to change this later
            continue

        # matrix for a single day and calculate the times for it
        day_matrix = pd.DataFrame(0.0, index=unique_crossings, columns=time_intervals)
        for _, row in day_data.iterrows():
            crossing_id = row["CrossingID"]
            start, end = row["BlockageStart"], row["BlockageClear"]

            curr_time = start.floor("5T")
            while curr_time < end:
                next_time = curr_time + pd.Timedelta(minutes=5)
                duration_seconds = (min(next_time, end) - max(curr_time, start)).total_seconds()

                if duration_seconds > 0:
                    time_slot_str = f"{curr_time.strftime('%H:%M')}-{next_time.strftime('%H:%M')}"
                    if time_slot_str in day_matrix.columns:
                        day_matrix.loc[crossing_id, time_slot_str] += duration_seconds
                curr_time = next_time

        cumulative_matrix += day_matrix # add the big matrix
        for crossing_id in day_data["CrossingID"].unique():
            day_counts[crossing_id] += 1

    # Avoid div by 0 (replace 0s with 1s) -- reset it later
    day_counts_safe = day_counts.replace(0, np.nan)
    averaged_matrix = cumulative_matrix.div(day_counts_safe, axis=0).fillna(0)
    blockage_numpy_matrix = averaged_matrix.to_numpy()
    averaged_matrix.to_csv(output_csv, index=True)
    return averaged_matrix, blockage_numpy_matrix


def merge_blockage_data():
    """
    Merges our blockage data.
    
    Inputs:
    - None

    Returns none.
    """
    # get both raw blockage datasets
    input_csv1 = os.path.join(BASE_DIR, "../raw_data/TRAINFO/HoustonBlockageData.csv")
    input_csv2 = os.path.join(BASE_DIR, "../raw_data/TRAINFO/HoustonBlockageData2.csv")
    output_csv = os.path.join(BASE_DIR, "../clean_data/merged_blockage_data.csv")
    
    # read as dataframes and merge them
    df1 = pd.read_csv(input_csv1)
    df2 = pd.read_csv(input_csv2)
    merged_df = pd.concat([df1, df2], ignore_index=True)

    merged_df.to_csv(output_csv, index=False)


# run our cleaning program
if __name__ == "__main__":
    # Get the base directory and the relative paths to get raw data and export the cleaned data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    merge_blockage_data()
    input_csv = os.path.join(BASE_DIR, "../clean_data/merged_blockage_data.csv")
    cleaned_csv = os.path.join(BASE_DIR, "../clean_data/blockage_data_with_duration.csv")

    # Run the preprocessing and check first 5 rows
    blockage_data_cleaned = preprocess_blockage_data(input_csv, cleaned_csv)
    #print(blockage_data_cleaned.head())

    # TEST: Run transformation for 5/22/23 between 1 AM - 11 PM
    transformed_csv = os.path.join(BASE_DIR, "../clean_data/transformed_blockage_data_5_22_23_1AM_11PM.csv")
    blockage_matrix, blockage_numpy_matrix = transform_blockage_matrix(blockage_data_cleaned, transformed_csv,
                                                                       start_time="01:00", end_time="23:00", target_day="2023-05-22")
    print(f"Blockage matrix shape: {blockage_numpy_matrix.shape}")
    print(blockage_matrix.head())

    # TEST: run the transformation for ALL days --
    # IMPORTANT: had to comment out because file may have been too large when combining both data sets!!!
    transformed_csv = os.path.join(BASE_DIR, "../clean_data/average_blockage_matrix.csv")
    # blockage_matrix, blockage_numpy_matrix = average_blockage_matrix(blockage_data_cleaned, transformed_csv,
    #                                                                     start_time="10:00", end_time="15:00")

    # print(f"Averaged matrix shape: {blockage_numpy_matrix.shape}")
    # print(blockage_matrix.head())
