import pandas as pd
import os
from collections import Counter
import csv
import re


def clean_dispatch_data(input_csv, output_csv):
    """
    Reads the 1st Arrival Performance raw dispatch data and adds basic additional features (columns).
    Additionally, it filters the data to only include the stations that are in east Houston.
    
    New columns:
    - Type (e.g., "F1", "F2", etc.)
    
    Inputs:
    - input_csv (str): Path to the raw dispatch data CSV.
    - output_csv (str): Path where the cleaned CSV will be saved.
    
    Returns:
    - dispatch_data_cleaned (pd.DataFrame): Cleaned dataset.
    """
    
    # Load the dispatch data
    dispatch_data = pd.read_csv(input_csv)
    
    # Define the mapping for 'new_cad'
    new_cad = {"F1": ['FFAA','FFHV','FFDM','FFCM'], 
             "F2": ['FFSB','FFMB'], 
             "F3": ['FFLB'], 
             "F4": ['FFHR'], 
             "E1": ['FEABA','FEASA','FEBAA','FEFAA','FEMAA','FESIA','FEUAA'], 
             "E2": ['FECPC','FEDIC','FEHTC','FEODC','FEREC','FESTC','FESYC'], 
             "E3": ['FECAD', 'FEMAD', 'FERED','FESHD']}
    
    # Create the 'new_cad' column based on 'cad_code_final'
    dispatch_data['new_cad'] = dispatch_data['cad_code_final'].apply(
        lambda x: next((k for k, v in new_cad.items() if x in v), None)
    )
    
    # Drop rows where 'new_cad' is None or NaN
    dispatch_data = dispatch_data.dropna(subset=['new_cad'])
    
    # Define the East Houston stations
    east_houston_stations = [1, 7, 8, 9, 12, 13, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 
                             34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 50, 52, 54, 56, 58, 61, 63, 64, 
                             65, 71, 72, 74, 80, 83, 86, 87, 88, 90, 91, 92, 93, 94]
    
    # Extract the station ID (numeric part) from the 'first_arriver_home_station' column
    def parse_station(entry):
        """
        Parse the station number from the 'Station X' format or numeric value.
        """
        # Check if the entry matches the 'Station X' format or is a numeric value
        if isinstance(entry, str) and re.search(r'Station (\d+)', entry):
            match = re.search(r'Station (\d+)', entry)
            return int(match.group(1)) if match else None
        elif isinstance(entry, str) and re.search(r'\d+', entry):
            # Handle numeric strings without the "Station" label
            return int(re.search(r'\d+', entry).group())
        elif isinstance(entry, (int, float)):
            return int(entry)  # Handle numeric entries directly
        return None

    # Apply the parsing function to 'first_arriver_home_station'
    dispatch_data['first_arriver_home_station'] = dispatch_data['first_arriver_home_station'].apply(parse_station)
    
    # Filter out invalid station IDs
    dispatch_data = dispatch_data[dispatch_data['first_arriver_home_station'].notnull()]
    
    # Filter for East Houston stations
    dispatch_data = dispatch_data[dispatch_data['first_arriver_home_station'].isin(east_houston_stations)]
    
    # Save the cleaned data to the output CSV
    dispatch_data.to_csv(output_csv, index=False)
    
    return dispatch_data



def station_resources_helper(input_file, output_file, col_name, east_houston_stations):
    """
    Processes the station data and creates a new CSV with the resources for each station in East Houston.
    
    Inputs:
    - input_file (str): Path to the input CSV.
    - output_file (str): Path where the cleaned CSV will be saved.
    - col_name (str): The column name that contains the assigned equipment data.
    - east_houston_stations (list): List of station IDs for East Houston.

    Returns none.
    """
    unique_words = set()

    # Collect all unique resources in the 'Assigned Equipment' column
    with open(input_file, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            words = [word.strip() for word in row[col_name].split(", ")]
            unique_words.update(words)

    # Sort unique words to ensure consistent column names
    unique_words = sorted(unique_words)

    # Create and write the new CSV with resource columns
    with open(input_file, newline="") as file, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(file)
        
        # Define the desired columns: Station and the resources of interest
        fieldnames = ['Station'] + ["Engine", "Ladder", "Medic", "Ambulance"]  # Only these columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            station_id = row["Station"]

            # Clean the Station ID to ignore non-numeric values or handle invalid entries
            try:
                # Remove non-numeric characters (if any)
                cleaned_station_id = re.sub(r'\D', '', station_id)  # This keeps only numeric characters

                # Check if the cleaned station ID is in the East Houston list
                if int(cleaned_station_id) not in east_houston_stations:
                    continue
            except ValueError:
                # If conversion fails (e.g., if the station_id is empty or contains non-numeric characters),
                # we skip this row.
                continue

            words = [word.strip() for word in row[col_name].split(", ")]
            word_counts = Counter(words)

            # Create a new row with the station ID and resource counts
            new_row = {'Station': row['Station']}
            new_row.update({
                "Engine": word_counts.get("Engine", 0),
                "Ladder": word_counts.get("Ladder", 0),
                "Medic": word_counts.get("Medic", 0),
                "Ambulance": word_counts.get("Ambulance", 0)
            })

            writer.writerow(new_row)

def fire_station_resources(input_csv, output_csv):
    """
    Reads the HFD Station information and creates a dataframe with the resources for each station in East Houston.
    
    Inputs:
    - input_csv (str): Path to the raw station information CSV.
    - output_csv (str): Path where the cleaned CSV will be saved.

    Returns:
    - station_resources (pd.DataFrame): Dataframe with the resources for each station.
    """
    
    # Define the East Houston station IDs
    east_houston_stations = [1,7,8,9,12,13,17,18,19,20,22,23,24,25,26,27,29,30,31,32,34,35,36,38,39,40,41,42,43,44,45,46,50,52,54,56,58,61,63,64,65,71,72,74,80,83,86,87,88,90,91,92,93,94]
    
    # Call the helper function to process the input file and filter for East Houston stations
    station_resources_helper(input_csv, output_csv, 'Assigned Equipment', east_houston_stations)
    
    # Now load the cleaned data back into a dataframe
    station_resources = pd.read_csv(output_csv)

    return station_resources


# run our cleaning program
if __name__ == "__main__":
    # Get the base directory and the relative paths to get raw data and export the cleaned data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Run the cleaning program for the dispatch data
    input_csv_1 = os.path.join(BASE_DIR, "../raw_data/1st Arrival Performance/01 2019 06 2019.csv")
    cleaned_csv_1 = os.path.join(BASE_DIR, "../clean_data/dispatch_data_cleaned.csv")
    dispatch_data_cleaned = clean_dispatch_data(input_csv_1, cleaned_csv_1)
    
    # Run the cleaning program for the station resources data
    input_csv_2 = os.path.join(BASE_DIR, "../raw_data/HFD Stations.csv")
    cleaned_csv_2 = os.path.join(BASE_DIR, "../clean_data/station_resources.csv")
    station_resources = fire_station_resources(input_csv_2, cleaned_csv_2)
