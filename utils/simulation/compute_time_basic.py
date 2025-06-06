import numpy as np
import pandas as pd
import datetime
from math import radians, cos, sin, asin, sqrt
from utils.simulation.check_crossing import check_crossing

def add_noise(distance_miles, emergency_coords):
    """
    Adds noise to the response time based on distance.
    The further the distance, the more uncertainty is introduced.

    Input:
    - distance_miles (int): distance we want to take into account 

    Returns the added noise.
    """
    # get the mean (~ 3 mins of delay per 60 miles for right now)
    noise_mean = 0.05 * distance_miles 
    noise_std = 0.02 * distance_miles

    lat, long = emergency_coords # TODO: grab coords and use them to compute noise

    # TODO: maybe add a parameter that checks where we are going 
    # and change noise accordingly (urban vs rural areas / time of day)
    noise = np.random.normal(loc=noise_mean, scale=noise_std)
    return max(0, noise)

def compute_time(check_crossing, emergency_obj, stations_list):
    """
    Computes response times for an emergency.

    Inputs:
    - check_crossing (function): A function to check crossing blockages.
    - emergency_obj: An instance of the Emergency class with 'emergency_time', 
      'emergency_type', and 'emergency_loc'.
    - stations_list: A list of station IDs
    
    Returns:
    - total_response_time (dict): A dictionary mapping station indices to response times (float).
    - blockages (dict): The dictionary maps each station_id to a tuple of two ordered lists: the 
                        first lists the sequence of crossings the unit passed, and the second has 
                        1s (blockage) and 0s (no blockage) for each crossing.
    """
    time = emergency_obj["time"].time()
    date = emergency_obj["time"].date()
    check_crossing_input = check_crossing(time, date)
    crossing_ids = check_crossing_input["CrossingID"].tolist()
    
    speed_mph = 60

    full_station_data = pd.read_csv(r'./clean_data/full_station_data.csv')

    total_response_time = {}
    blockages = {}

    def haversine_miles(coord1, coord2):
      """
      Inner helper function that calculates the distance between two coordinates
      using the haversine formula.

      Inputs:
      - coord1, coord2 (each a tuple): contains a latitude and a longitude.

      Returns the distance in miles.
      """
      # Coordinates in decimal degrees
      lat1, lon1 = coord1
      lat2, lon2 = coord2

      # Convert to radians
      lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

      # Haversine formula
      dlat = lat2 - lat1 
      dlon = lon2 - lon1 
      a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
      c = 2 * asin(sqrt(a)) 
      r = 3958.8  # Radius of Earth in miles
      return c * r  # Distance in miles
    
    for station in stations_list:
        station_row = full_station_data[full_station_data["station_id"] == station]
        station_loc = (station_row["Latitude"].values[0], station_row["Longitude"].values[0])
        emergency_coord = emergency_obj["coordinates"]  # (lat, lon)

        distance_miles = haversine_miles(station_loc, emergency_coord)
        time_hr = distance_miles / speed_mph
        time_min = time_hr * 60

        # generate noise based on calculated distance and add it to our total response time
        noise = add_noise(distance_miles, emergency_coord)
        total_response_time[station] = (time_min, noise)
        # blockages[station] = check_crossing_input(emergency_coord, station_loc)
  
    # random blockages
    for station_i, station in enumerate(stations_list):
        num_crossings = 1 + (station_i % 3) # 1, 2, or 3
        passed_crossings = np.random.choice(crossing_ids, num_crossings)
        blockages[station] = (list(passed_crossings), [1 if (j%2 == 0) else 0 for j in range(num_crossings)])

    return total_response_time, blockages
    

# Testing
if __name__ == "__main__":

  # Emergency object instance
  emergency_obj = {}
  emergency_obj["time"] = datetime.datetime(2024, 5, 3, 10, 41, 15)
  emergency_obj["emergency_type"] = "E1"
  emergency_obj["coordinates"] = (29.70223, -95.5393)

  # A subset of station ids we are interested in
  station_data = pd.read_csv(r'./clean_data/full_station_data.csv')
  stations_list = list(station_data["station_id"])

  # compute time 
  total_response_time, blockages = compute_time(check_crossing, emergency_obj, stations_list)
  print(total_response_time)
  print(blockages)