import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

from utils.simulation.simulation import Simulator
from utils.dispatch import dispatch
from utils.simulation.compute_time import compute_time
from utils.simulation.check_crossing import check_crossing


def test_instance(date):
    """
    Filters and returns emergency data based on the specified date.

    Input:
    - date (str): date in "YYYY-MM-DD" format

    Returns:
    - units (set): A set of unit types (e.g., Engine, Ladder, Medic, Ambulance).
    - stations (dict): A dictionary mapping station IDs to their coordinates (latitude, longitude).
    - supply (dict): A dictionary mapping station IDs to a list of available units.
    - emergency_arrivals (pd.DataFrame): A DataFrame containing emergency data with the following columns:
        - "emergency_type" (str): The type of emergency.
        - "time" (datetime): The timestamp of the emergency.
        - "coordinates" (tuple): The latitude and longitude of the emergency location.
    """
    dir = os.path.dirname(os.path.abspath(__file__))

    def get_clean_data(filename):
        """ Helper that given a filename string, returns the appropiate path."""
        path = os.path.join(dir, "..", "..", "clean_data", filename)
        return os.path.normpath(path)

    full_station_data = pd.read_csv(get_clean_data("full_station_data.csv"))

    # Stations and units remain constant
    units = {0, 1, 2, 3}  # engine, ladder, medic, ambulance
    stations = {row["station_id"]: (row["Latitude"], row["Longitude"]) for _, row in full_station_data.iterrows()}

    # at the start of the day, we have a full supply of availability
    supply = {row["station_id"]: [row["Engine"],
                                  row["Ladder"],
                                  row["Medic"],
                                  row["Ambulance"]] for _, row in full_station_data.iterrows()}


    # go through the dispatch data and create a df on emergency arrivals
    col_names = ["occurred_on_local_date", "occurred_on_local_time", "latitude", "longitude", "new_cad"]
    emergency_arrivals = pd.read_csv(get_clean_data("dispatch_data_cleaned.csv"))[col_names]
    emergency_arrivals.columns = ["date", "time", "lat", "long", "type"]

    # Filter data by the given date
    emergency_arrivals["date"] = pd.to_datetime(emergency_arrivals["date"], format="%m/%d/%Y")
    filtered_emergencies = emergency_arrivals[emergency_arrivals["date"] == date]
    emergency_arrivals = pd.DataFrame({"emergency_type": filtered_emergencies["type"],
                                       "time": [datetime.strptime(f"{row['date'].strftime('%m/%d/%Y')} {row['time']}", "%m/%d/%Y %H:%M:%S")
                                                for _, row in filtered_emergencies.iterrows()],
                                       "coordinates": [(float(row["lat"]), float(row["long"]))
                                                       for _, row in filtered_emergencies.iterrows()]}).sample(frac=0.5, random_state=42)
    emergency_arrivals = emergency_arrivals.sort_values(by="time")
    return units, stations, supply, emergency_arrivals


def greedy_cost(times, blockages, check_crossing, emergency, stations):
    """
    Computes the basic greedy cost (dispatch time) for each station, ignoring blockages.
    There is only one used input -- ignore the others. (should be changed). Used in the MIP.

    Inputs:
    - times (dict[int, tuple[float, float]]): maps station indices to (dispatch time, noise) tuples.

    Returns:
    - list[float]: List of dispatch times (in minutes), sorted by station index.
    """
    # time.sleep(5)
    # times[k][0] because times[k] is a tuple (the time, the noise) in minutes.
    # return [times[k][0] for k in sorted(stations.keys())]
    return [times[station][0] for station in sorted(times.keys())]


def greedy_block_cost(times, blockages, check_crossing, emergency, stations):
    """
    Computes a greedy cost that adds expected delay due to blockages at railroad crossings. Used in the MIP.

    Inputs:
    - times (dict[int, tuple[float, float]]): maps station indices to (dispatch time, noise) tuples.
    - blockages (dict[int, tuple[list[int], list[int]]]): maps station indices to (crossing IDs, blockage flags).
    - check_crossing (func): function that returns predicted blockage status for crossings given a datetime.
    - emergency (pd.Series): series containing 'time' field (datetime) for the emergency.
    - stations (dict[int, tuple[float, float]]): maps station indices to their coordinates.

    Returns:
    - list[float]: List of adjusted dispatch times (including blockage delays), sorted by station index.
    """
    blockage_data = pd.read_csv(r'./clean_data/blockages_gdf_avgd_MORE_COLS.csv')

    def add_time(crossings, status, blockage):
        """Estimates total delay time caused by railroad crossings."""
        if len(crossings) == 0:
            return 0
        else:
            total = 0
            for i in range(len(crossings)):
                crossing = crossings[i]
                row = blockage.loc[blockage['CrossingID'] == crossing].iloc[:, 1:].values.tolist()[0]
                mean_time = blockage_data.loc[blockage_data['CrossingID'] == crossings[i]]['BlockageTime_mean'].item()
                if 1 in row:
                    total += random.choices([mean_time, 0], weights=[0.9, 0.1])[0]
                else:
                    total += random.choices([0, mean_time], weights=[0.9, 0.1])[0]
            return total

    emergency_time = emergency['time']
    blockage = check_crossing(emergency_time.time(), emergency_time.date())
    return [times[k][0] + add_time(blockages[k][0], blockages[k][1], blockage) for k in sorted(stations.keys())]


def run_simulation(date, method_a, list_travel_times=None, list_paths=None):
    """
    Runs the simulation for a specific date and method and prints the relevant results.

    Inputs:
    - date (str): date in "YYYY-MM-DD" format
    - method_a (bool): true if we want to run the naive method_a, false if we want to run method_b.

    Returns:
    - total_response_time (float): the total response time across all emergencies
    - job_logs (List): list of job logs
    - blockages (list): list of blockages
    """
    date_instance = test_instance(date)
    units, stations, supply, emergency_arrivals = date_instance

    # METHOD A: will have a greedy_cost function
    if method_a:
        sim = Simulator(units, stations, supply, dispatch.solve_assignment, greedy_cost, compute_time)

        total_response_time, job_logs, blockages = sim.process_assignment(emergency_arrivals, list_travel_times, list_paths)
    # METHOD B: will have a greedy_block_cost function
    else:
        sim = Simulator(units, stations, supply, dispatch.solve_assignment, greedy_block_cost, compute_time)

        total_response_time, job_logs, blockages = sim.process_assignment(emergency_arrivals, list_travel_times, list_paths)

    print("Total time = " + str(total_response_time) +"\n assignments =" + str(job_logs) + "\n blockages = " + str(blockages))
    return total_response_time, job_logs, blockages


def print_summary_statistics():
    """
    Prints summary statistics comparing the two simulation methods.
    Returns none.
    """
    
    months = ['jan', 'feb', 'mar', 'apr', 'may']
    results = []
    for month in months:
        with open(f'{"results"}/{month}_results.pkl', 'rb') as file:
             results.append(pickle.load(file))

    sum1, count, sum2, sum3 = 0, 0, 0, 0
    for result in results:
        for i in range(len(result[0][1])):
            if result[1][1][i][1] != result[0][1][i][1]:
                count += 1
                sum2 += result[1][1][i][-1]
                sum3 += result[0][1][i][-1]
                sum1 += (result[0][1][i][-1] - result[1][1][i][-1]) / result[1][1][i][-1]

    for i in range(len(months)):
        sum1, sum2, count = 0, 0, 0
        result = results[i]
        for j in range(len(result[0][1])):
            count += 1
            sum1 += result[1][1][i][-1]
            sum2 += result[0][1][i][-1]
        print(f'{months[i].capitalize()} Average (Geographically Closest):', sum1 / count)
        print(f'{months[i].capitalize()} Average (Avoids Crossings):', sum2 / count)
        print()

    print('Percent Improvement:')
    print(sum1 / count * 100)
    print('Average Response Time (Geographically Closest):')
    print(sum2 / count)
    print('Average Response Time (Avoids Crossings):')
    print(sum3 / count)
    print('Number of Cases Compared:')
    print(count)
    print()


# testing purposes
if __name__ == "__main__":

    # open pre-ran simulation results from our results folder
    with open('results/jan_results.pkl', 'rb') as file:
        jan_results = pickle.load(file)
    with open('results/feb_results.pkl', 'rb') as file:
        feb_results = pickle.load(file)
    with open('results/mar_results.pkl', 'rb') as file:
        mar_results = pickle.load(file)
    with open('results/apr_results.pkl', 'rb') as file:
        apr_results = pickle.load(file)
    with open('results/may_results.pkl', 'rb') as file:
        may_results = pickle.load(file)


    # plot our results
    results = [jan_results, feb_results, mar_results, apr_results, may_results]
    sum1, count, sum2, sum3 = 0, 0, 0, 0
    for result in results:
        for i in range(len(result[0][1])):
            vals1 = [x for sublist in result[0][1][i][1].values() for x in sublist]
            vals2 = [x for sublist in result[1][1][i][1].values() for x in sublist]
            if result[1][1][i][1] != result[0][1][i][1]:
                count += 1
                sum2 += result[1][1][i][-1]
                sum3 += result[0][1][i][-1]
                sum1 += (result[0][1][i][-1] - result[1][1][i][-1]) / result[1][1][i][-1]
    print('percent')
    print(sum1 / count * 100)
    print('avg greedy')
    print(sum2 / count)
    print('avg ours')
    print(sum3 / count)
    print(count)
    print()
    freqs1 = []
    freqs2 = []
    for result in results:
        freqs1 += [result[0][1][i][-1] for i in range(len(result[0][1]))]
        freqs2 += [result[1][1][i][-1] for i in range(len(result[0][1]))]

    sns.kdeplot(freqs1, color='blue', fill=True, alpha=0.5, label='Avoids Crossings')
    sns.kdeplot(freqs2, color='orange', fill=True, alpha=0.5, label='Geographically Closest')

    plt.axvline(np.average(freqs1), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(np.average(freqs2), color='orange', linestyle='dashed', linewidth=2)
    print(np.average(freqs1))
    print(np.average(freqs2))

    plt.title("Density Estimate of Response Times", fontsize=20)
    plt.xlabel("Response Time (minutes)", fontsize=18)
    plt.ylabel("Relative Frequency", fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

    plt.close()

    sns.ecdfplot(freqs1, color='blue', alpha=0.5, label='Avoids Crossings')
    sns.ecdfplot(freqs2, color='orange', alpha=0.5, label='Geographically Closest')
    plt.title("Cumulative Density Estimate of Response Times", fontsize=20)
    plt.xlabel("Response Time (minutes)", fontsize=18)
    plt.ylabel("Cumulative Relative Frequency", fontsize=18)
    plt.legend(fontsize=18)
    plt.show()
