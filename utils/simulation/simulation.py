import os
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import heapq
import random
from utils.dispatch import dispatch
from utils.simulation.compute_time import compute_time
from utils.simulation.check_crossing import check_crossing

class Simulator:
	def __init__(self, units, stations, supply, assignment_func, cost_func, travel_time_func):
		"""
		This constructor initializes an instance of the Simulator object.

		Inputs:
		- units (set[int]): A set of indices representing different types of emergency vehicles 
		(e.g., engine, ladder, medic, ambulance).
		- stations (dict[int, tuple[float, float]]): A dictionary mapping station indices to their geographic coordinates.
		Each key-value pair (j, (l1, l2)) represents station j at location (latitude l1, longitude l2).
		- supply (dict[List[int]]): A dictionary mapping station indices to what vehicles are available.
		Each key-value pair (j, L) represents station j having the vehicles in the list L, where L[i] denotes how many
		units of vehicle type i station j has.
		- assignment_func (function): Function that assigns vehicles from stations.
		- cost_func(function): Function that computes the costs to be used in the MIP.
		- travel_time_func (function): Function that calculates travel times between the station and the emergency.

		Returns:
		- An instance of the Simulator class.
		"""
		self.units = units
		self.stations = stations
		self.supply = supply
		self.assignment_func = assignment_func
		self.cost_func = cost_func
		self.travel_time_func = travel_time_func
		self.busy_until = {station: {vtype: [] for vtype in range(len(next(iter(supply.values()))))}
		                   for station in supply.keys()}

	def process_assignment(self, emergency_arrivals, list_travel_times=None, list_paths=None):
		"""
		Runs a simulation of emergency responses, calculates the total response time, and logs the response time 
		and assignment details for each emergency.

		Inputs:
		- emergency_arrivals (List[emergency_obj]): a list of
		emergency_objs, which contain the time of the emergency, the location of the emergency, and
		the type of the emergency.
		- blockages (List[matrix]): a list of matrices of binary integers, where a 1 represents there being a
		blockage and 0 represents whether there's a blockage at a certain crossing. Each entry of the list
		corresponds to a specific emergency, and each matrix corresponds to the crossings that are blocked at
		the time of the emergency for ten minutes afterward.

		Returns:
		- total_response_time (float): the total response time across all emergencies
		- job_logs (List): list of job logs
		"""
		random.seed(1)
		# Initialization of total response time and details of each emergency
		total_response_time = 0
		job_logs = []
		blockage_count = 0
		blockage_data = pd.read_csv(r'./clean_data/blockages_gdf_avgd_MORE_COLS.csv')

		# TOP_FOLDER = "../../clean_data"
		TOP_FOLDER = "./clean_data"
		SHAPEFILES_FOLDER = "All_Shapefiles_in_East_Houston"
		stations_gdf = gpd.read_file(TOP_FOLDER + "/" + SHAPEFILES_FOLDER + "/fire_stations_gdf_East")
		crossings_gdf = gpd.read_file(TOP_FOLDER + "/" + SHAPEFILES_FOLDER + "/blockages_gdf_avgd_East")

		# Iterate through the arrivals for the day
		e_iter = 0
		num_emer = len(emergency_arrivals)
		for i, (_, emergency) in enumerate(emergency_arrivals.iterrows()):

			emergency_time = emergency["time"]
			blockage = check_crossing(emergency_time.time(), emergency_time.date())
			emergency_minutes = 60*emergency_time.time().hour + emergency_time.time().minute

			emergency_type = emergency["emergency_type"]
			emergency_loc = emergency["coordinates"]

			# Replace with how long it takes to alleviate the emergency later
			response_time = max([0, random.gauss(20, 5)])

			curr_supply = {j: [self.supply[j][i] - len(self.busy_until[j][i]) for i in range(len(self.units))]
			               for j in self.stations.keys()}

			# Get assignment from dispatch MIP
			print(f"Starting to process Emergency #{e_iter} out of {num_emer} at time {emergency_time}\n")
			if list_travel_times is None and list_paths is None:
				travel_times, paths = compute_time(check_crossing, emergency, self.stations)
			else:
				travel_times = list_travel_times[i]
				paths = list_paths[i]
			
			costs = self.cost_func(travel_times, paths, check_crossing, emergency, self.stations)
			
			# travel_times, paths = compute_time(check_crossing, emergency, self.stations)
			# costs = self.cost_func(travel_times, paths, check_crossing, emergency, self.stations)
			
			# Get assignment from dispatch MIP
			vehic_assignments = self.assignment_func(self.units, self.stations, emergency_type, costs, curr_supply)

			# Get travel time from the time the emergency was called to the last vehicle to arrive
			response_completion_times = []

			# Iterate through the vehicle assignments
			for vehic_type, assignments in vehic_assignments.items():
				for station in assignments:
					# Calculate the travel time for the vehicle
					# travel_times, paths = compute_time(check_crossing, emergency, list(self.stations.keys()))
					
					if travel_times[station][0] is None:
						continue

					travel_time = travel_times[station][0] # (time, noise)
					crossings, truths = paths[station]

					for l in range(len(crossings)):
						# Change this condition later
						row = blockage.loc[blockage['CrossingID'] == crossings[l]].iloc[:, 1:].values.tolist()[0]
						mean_blockage_time = blockage_data.loc[blockage_data['CrossingID'] == crossings[l]]['BlockageTime_mean'].item()
						std_blockage_time = blockage_data.loc[blockage_data['CrossingID'] == crossings[l]]['BlockageTime_std'].item()
						if 1 in row and truths[l] == 1:
							# Replace with random blockage time (time it takes to take care of emergency)
							travel_time += -np.log(random.random()) * mean_blockage_time + mean_blockage_time - std_blockage_time
							blockage_count += 1

					# Calculates total response time if the number of vehicles of a certain type being used is less
					# than the supply
					if len(self.busy_until[station][vehic_type]) < curr_supply[station][vehic_type]:
						start_time = emergency_minutes

					# Otherwise, queue the vehicle
					else:
						start_time = max(heapq.heappop(self.busy_until[station][vehic_type]), emergency_minutes)

					response_duration = travel_time + response_time
					completion_time = start_time + response_duration
					heapq.heappush(self.busy_until[station][vehic_type], completion_time)
					response_completion_times.append(completion_time)

			for j in self.stations.keys():
				for i in range(len(self.units)):
					old_queue = self.busy_until[j][i]
					new_queue = [t for t in old_queue if t >= emergency_minutes]
					curr_supply[j][i] += len(old_queue) - len(new_queue)
					self.busy_until[j][i] = new_queue

			# Get the final response time plus the travel time
			emergency_finish_time = max(response_completion_times)
			response_duration = emergency_finish_time - emergency_minutes

			# Calculate total response time
			total_response_time += response_duration

			# Add entry to job logs
			job_logs.append((emergency_time, vehic_assignments, response_duration))

			e_iter += 1

		return total_response_time, job_logs, blockage_count


# early tests before creating the run_simulation file
if __name__ == "__main__":
	dir = os.path.dirname(os.path.abspath(__file__))

	def get_clean_data(filename):
		path = os.path.join(dir, "..", "..", "clean_data", filename)
		return os.path.normpath(path)

	# Import available units at each station
	full_station_data = pd.read_csv(get_clean_data("full_station_data.csv"))
	stations = {row["station_id"]: (row["Latitude"], row["Longitude"]) for _, row in full_station_data.iterrows()}
	supply = {row["station_id"]: [row["Engine"],
	                              row["Ladder"],
	                              row["Medic"],
	                              row["Ambulance"]] for _, row in full_station_data.iterrows()}

	# Import test data
	col_names = ["occurred_on_local_date", "occurred_on_local_time", "latitude", "longitude", "new_cad"]
	emergency_arrivals = pd.read_csv(get_clean_data("dispatch_data_cleaned.csv"))[col_names]
	emergency_arrivals.columns = ["date", "time", "lat", "long", "type"]

	emergency_arrivals = pd.DataFrame({"emergency_type": emergency_arrivals["type"],
	                                   "time": [datetime.strptime(f"{row['date']} {row['time']}", "%m/%d/%Y %H:%M:%S")
	                                            for _, row in emergency_arrivals.iterrows()],
	                                   "coordinates": [(float(row["lat"]), float(row["long"]))
	                                                   for _, row in emergency_arrivals.iterrows()]})
	emergency_arrivals = emergency_arrivals.sort_values(by="time")

	units = {0, 1, 2, 3}

	def greedy_cost(check_crossing, emergency, stations):
		times, blockages = compute_time(check_crossing, emergency, stations)
		return [times[k] for k in sorted(stations.keys())]


	sim = Simulator(units, stations, supply, dispatch.solve_assignment, greedy_cost, compute_time)
	total_response_time, job_logs = sim.process_assignment(emergency_arrivals, [])
	print(total_response_time)
