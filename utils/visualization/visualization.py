import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import ast
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation
import pickle
import seaborn as sns

def plot_dispatch_map(stations, emergency_data, station_to_emergency_map):
    """
    Plots the dispatch of emergency vehicles from fire stations to emergencies with animated lines.

    Inputs:
    - stations (dict): maps station id to a tuple of coordinates (latitude, longitude)
    - emergency_data (list): list of emergency locations, each entry is a tuple of coordinates
    - station_to_emergency_map (dict): maps station IDs to emergency locations
    
    Returns none.
    """
    fig, ax = plt.subplots(figsize=(10, 10))  # Create figure and axis

    background_image = r'./clean_data/blockage_maps/houston_districts_with_blockages_cropped.png'
    img = mpimg.imread(background_image)

    xmin, xmax = -95.85, -95.02  # Longitude range
    ymin, ymax = 29.45, 30.22    # Latitude range

    # Set the axis limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Overlay the image as background
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="upper", aspect="auto", alpha=1)

    # Plot stations
    for station_id, (lat, lon) in stations.items():
        ax.scatter(lon, lat, c='blue', label='Station' if station_id == list(stations.keys())[0] else "")
    
    # Initialize empty lines and persistent emergency markers for animation
    lines = []
    persistent_markers = []  # List to store markers that should remain on the graph
    assignments = list(station_to_emergency_map.items())

    # Animation function
    def update(frame):
        current_assignment_index = frame // 100  # Each assignment takes 100 frames
        progress = frame % 100  # Progress within the current assignment

        # Draw the current emergency marker
        if current_assignment_index < len(assignments):
            station_id, emergency_loc = assignments[current_assignment_index]
            if station_id in stations and emergency_loc in emergency_data:
                # Add the emergency marker to the persistent list if it's not already there
                if len(persistent_markers) <= current_assignment_index:
                    emergency_marker = ax.scatter(emergency_loc[1], emergency_loc[0], c='red', marker='*', s=150)
                    persistent_markers.append(emergency_marker)

                # Draw the line for the current assignment
                station_loc = stations[station_id]
                x = [station_loc[1], station_loc[1] + (emergency_loc[1] - station_loc[1]) * progress / 100]
                y = [station_loc[0], station_loc[0] + (emergency_loc[0] - station_loc[0]) * progress / 100]

                if len(lines) <= current_assignment_index:
                    line, = ax.plot([], [], linestyle=':', color='red', alpha=0.85, linewidth=2.5)
                    lines.append(line)
                lines[current_assignment_index].set_data(x, y)

        return lines + persistent_markers

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(assignments) * 100, interval=50, blit=True, repeat=False
    )

    # Manually create legends
    station_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Station')
    emergency_legend = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=10, label='Emergency')
    assignment_line = mlines.Line2D([], [], color='red', linestyle=':', label='Assignment')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=[station_legend, emergency_legend, assignment_line], loc='upper left')
    ax.set_title("Dynamic Emergency Vehicle Dispatch Map")
    ax.grid(True)

    plt.show()


def run_demo_dispatch_map():
    """
    Displays a dispatch map of emergency vehicles from fire stations to
    emergencies. 
    Inputs:
    - None

    Returns none.
    """
    # Load station data
    stations_df = pd.read_csv(r'./clean_data/fire_station_locs.csv')
    stations = stations_df.set_index("Station")[['Latitude', 'Longitude']].T.to_dict("list")

    if 52 in stations:
        del stations[52]

    # Load emergency data
    emergency_df = pd.read_csv(r'./clean_data/test.csv')
    emergency_df["emergency_loc"] = emergency_df["emergency_loc"].apply(ast.literal_eval)
    emergency_data = emergency_df["emergency_loc"].sample(n=10).tolist()

    # Create a dynamic mapping of stations to emergencies
    station_ids = list(stations.keys())
    station_to_emergency_map = {
        station_ids[i % len(station_ids)]: emergency_data[i] for i in range(len(emergency_data))
    }

    # Test the visualization function
    plot_dispatch_map(stations, emergency_data, station_to_emergency_map)



# need to globally append our results so I can call it the plots in our notebook (I don't want inputs in our plot functions)
months = ['jan', 'feb', 'mar', 'apr', 'may']
results = []
for month in months:
	with open(f'{"results"}/{month}_results.pkl', 'rb') as file:
		results.append(pickle.load(file))

def plot_density_estimate():
    """
    Plots a density estimate of response times for both methods.
    Returns none.
    """
    freqs1, freqs2 = [], []

    for result in results:
        freqs1 += [result[0][1][i][-1] for i in range(len(result[0][1]))]
        freqs2 += [result[1][1][i][-1] for i in range(len(result[0][1]))]

		
    plt.figure(figsize=(12,8))
    sns.kdeplot(freqs1, color='blue', fill=True, alpha=0.5, label='Method B: Avoids Crossings')
    sns.kdeplot(freqs2, color='orange', fill=True, alpha=0.5, label='Method A: Geographically Closest')

    plt.axvline(np.average(freqs1), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(np.average(freqs2), color='orange', linestyle='dashed', linewidth=2)

    print('Average Avoids Crossings:', np.average(freqs1))
    print('Average Geographically Closest:', np.average(freqs2))

    plt.title("Density Estimate of Response Times", fontsize=20)
    plt.xlabel("Response Time (minutes)", fontsize=18)
    plt.ylabel("Relative Frequency", fontsize=18)
    plt.legend(fontsize=14)
    plt.show()

# to just run locally without demo just run the file script
if __name__ == "__main__":
    run_demo_dispatch_map()