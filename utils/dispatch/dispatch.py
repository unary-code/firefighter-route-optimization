import gurobipy as gp
from geopy.distance import geodesic

def solve_assignment(units, stations, emergency, costs, avail_units):
	"""
	Solves the assignment problem for dispatching multiple types of emergency vehicles.

	Inputs:
	- units (set[int]): A set of indices representing different types of emergency vehicles 
	(e.g., engine, ladder, medic, ambulance).
	- stations (dict[int, tuple[float, float]]): A dictionary mapping station indices to their geographic coordinates.
	Each key-value pair (j, (l1, l2)) represents station j at location (latitude l1, longitude l2).
	- emergency (int): An integer representing the type of emergency, which determines the required vehicle types.
	- costs (list[float]): A list of size J (where J is the number of stations), containing (cost/time of dispatching, noise).
	- avail_units (dict[list[int]]): A dictionary mapping station indices to what vehicles are available.
	Each key-value pair (j, L) represents station j having the vehicles in the list L, where L[i] denotes how many
	units of vehicle type i station j has.

	Returns:
	- dict[int, list[int]]: A dictionary mapping vehicle_type -> [station_number for number_of_units_dispatched].
	Each key i represents vehicle type i being dispatched, where the value contains station number j for however 
	many vehicles are being assigned from that station.
	"""

	# Initialization of variables
	solutions = {}
	I = len(units)
	station_inds = list(stations.keys())
	J = len(station_inds)

	# Create a model

	# Initialize model
	model = gp.Model()
	model.params.LogToConsole = 0

	# Initialize decision variables
	x = model.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0)

	# Set model objective
	model.setObjective(sum(sum(costs[j] * x[i, j] for j in range(J)) for i in range(I)))

	# Capacity constraint
	for i in range(I):
		for j in range(J):
			model.addLConstr(x[i, j] <= avail_units[station_inds[j]][i])

	# Demand constraints
	if emergency == "F1":
		model.addLConstr(sum(x[0, j] for j in range(J)) + sum(x[1, j] for j in range(J)) >= 1)
	elif emergency == "F2":
		model.addLConstr(sum(x[0, j] for j in range(J)) >= 3)
		model.addLConstr(sum(x[1, j] for j in range(J)) >= 1)
	elif emergency == "F3":
		model.addLConstr(sum(x[0, j] for j in range(J)) >= 4)
		model.addLConstr(sum(x[1, j] for j in range(J)) >= 2)
		model.addLConstr(sum(x[2, j] for j in range(J)) + sum(x[3, j] for j in range(J)) >= 1)
	elif emergency == "F4":
		model.addLConstr(sum(x[0, j] for j in range(J)) >= 4)
		model.addLConstr(sum(x[1, j] for j in range(J)) >= 3)
		model.addLConstr(sum(x[2, j] for j in range(J)) + sum(x[3, j] for j in range(J)) >= 1)
	elif emergency == "E1":
		model.addLConstr(sum(x[0, j] for j in range(J)) + sum(x[1, j] for j in range(J)) +
		                 sum(x[3, j] for j in range(J)) >= 1)
	elif emergency == "E2":
		model.addLConstr(sum(x[2, j] for j in range(J)) >= 1)
	else:
		model.addLConstr(sum(x[0, j] for j in range(J)) + sum(x[1, j] for j in range(J)) >= 1)
		model.addLConstr(sum(x[2, j] for j in range(J)) >= 1)

	# Optimize
	model.optimize()

	# Collect nonzero portions of the solution
	for i in range(I):
		solutions[i] = []
		for j in range(J):
			if x[i, j].X != 0:
				solutions[i] += [station_inds[j] for _ in range(int(x[i, j].X))]

	return solutions


def greedy_assignment(units, stations, emergency_type, emergency_loc, avail_units):
	"""
	Greedily assigns emergency units from stations based on distance to the emergency location.

	Inputs:
	- units (set[int]): A set of indices representing different types of emergency vehicles 
	(e.g., engine, ladder, medic, ambulance).
	- stations (dict[int, tuple[float, float]]): A dictionary mapping station indices to their geographic coordinates.
	Each key-value pair (j, (l1, l2)) represents station j at location (latitude l1, longitude l2).
	- emergency_type (int): An integer representing the type of emergency, which determines the required vehicle types.
	- emergency_loc (tuple[float, float]): A pair (l1, l2) representing the latitude and longitude of the emergency.
	- avail_units (list[list[int]]): A matrix of size I x J, where I is the number of vehicle types and J is the number
	of stations. The value at avail_units[i][j] represents the number of available units of vehicle type i at station j.

	Returns:
	- dict[int, list[int]]: A dictionary mapping vehicle_type → [station_number for number_of_units_dispatched].
	Each key i represents vehicle type i being dispatched, where the value contains station number j for however 
	many vehicles are being assigned from that station.
	"""
	costs = []
	station_inds = list(stations.keys())
	for j in range(len(station_inds)):
		station_loc = stations[station_inds[j]]
		costs.append(geodesic(station_loc, emergency_loc).kilometers)

	return solve_assignment(units, stations, emergency_type, costs, avail_units)




