import math
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, MultiPoint, LineString
from polyline import decode as decode_polyline
from environs import Env
import matplotlib.pyplot as plt
from pyproj import Transformer

KM_TO_MILE = 0.621371
TOP_FOLDER = "./clean_data"
SHAPEFILES_FOLDER = "All_Shapefiles_in_East_Houston"
FILE_NAME = "rails_gdf_within_East_districts_COMP_best"

districts_gdf = gpd.read_file(TOP_FOLDER + "/" + SHAPEFILES_FOLDER + "/fire_districts_gdf_East")
rails_gdf_within_East_districts = gpd.read_file(TOP_FOLDER + "/" + SHAPEFILES_FOLDER + "/" + FILE_NAME + "/" + FILE_NAME + ".shp")

# Transform EPSG:4326 (lat/lon) to an appropriate projected CRS for accurate distance calculations
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
reverse_transformer_3395_to_4326 = Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)
transformer_4326_to_3395 = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)

def add_noise(time_mins):
    """
    Adds noise to the response time based on distance.
    The further the distance, the more uncertainty is introduced.

    Input:
        - time_mins (float): time (in minutes) we want to take into account 

    Returns:
        - noise(float): the added noise.
    """
    if (time_mins is None) or (np.isnan(time_mins)):
        return 0
    # get the mean (~ 3 mins of delay per 60 miles for right now)
    noise_mean = 0.05 * time_mins 
    noise_std = 0.02 * time_mins

    # TODO: maybe add a parameter that checks where we are going 
    # and change noise accordingly (urban vs rural areas / time of day)
    noise = np.random.normal(loc=noise_mean, scale=noise_std)
    return noise

def get_district_ind_of_point(point):
    """
    Calculates the index in "districts_gdf" of the district that a point resides in.

    Inputs:
        - point (shapely.geometry.Point): a point on a map.

    Returns:
        - int: the index in "districts_gdf" of the district that that point resides in.
    """
    containing_districts = districts_gdf[districts_gdf.geometry.contains(point)]
    
    if len(containing_districts) > 0:
        return containing_districts.index[0]
    else:
        return -1


def solve_startp(ip_3395, m, standard_length):
    """
    Solves for startp.x given ip in CRS 3395, slope m, and standard_length.

    Inputs:
        - ip_3395 (Point): the point in CRS 3395, around which we should extend the line.
        - m (float): slope.
        - standard_length (float): the length in meters.
    
    Returns:
        - pt1 (Point), pt2 (Point): two points both in CRS 4326, corresponding to the two endpoints of the extended line.
    """
    
    offset = standard_length / (2 * math.sqrt(1 + m**2))

    # Two possible solutions for startp.x
    x1 = ip_3395.x + offset
    x2 = ip_3395.x - offset
    pt1 = Point(x1, m*(x1 - ip_3395.x) + ip_3395.y)
    pt2 = Point(x2, m*(x2 - ip_3395.x) + ip_3395.y)
    pt1 = reverse_transformer_3395_to_4326.transform(pt1.x, pt1.y)
    pt2 = reverse_transformer_3395_to_4326.transform(pt2.x, pt2.y)
    return pt1, pt2

def extend_line_to_length(m, b, line_x, ip, rect_bounds=None, standard_length=None):
    """
    Extends a line with slope m and intercept b either to a fixed length or to the edges of a rectangle.

    Inputs: 
    - m (float or None): Slope of the line (None if vertical).
    - b (float): Y-intercept of the line.
    - line_x (float): Example x-value, only used if the line is vertical.
    - ip (Point): Intersection point on the line.
    - rect_bounds (tuple(float, float, float, float), optional): (min_x, min_y, max_x, max_y) rectangle bounds.
    - standard_length (float or int, optional): Desired line segment length in meters if no rectangle is given.

    Returns:
    - LineString: Extended line segment either fitted inside the rectangle or of fixed length.
    """

    if m is None:
        # The line is a vertical line
        if rect_bounds is None:
            ip_3395 = Point(transformer_4326_to_3395.transform(ip.x, ip.y))
            startp, endp = Point(ip_3395.x, ip_3395.y - standard_length/2.0), Point(ip_3395.x, ip_3395.y + standard_length/2.0)
            
            startp = reverse_transformer_3395_to_4326.transform(startp.x, startp.y)
            endp = reverse_transformer_3395_to_4326.transform(endp.x, endp.y)
            
            return LineString([startp, endp])
        else:
            return LineString([(line_x, rect_bounds[1]), (line_x, rect_bounds[3])])

    if abs(m) < 0.01:
        if rect_bounds is not None:
            return LineString([(rect_bounds[0], b), (rect_bounds[2], b)])
    
    if rect_bounds is None:
        # Extend to be "standard_length" meters
        ip_3395 = Point(transformer_4326_to_3395.transform(ip.x, ip.y))
        startp_1, startp_2 = solve_startp(ip_3395, m, standard_length)
        
        return LineString([startp_1, startp_2])
    else:
        # Extend to the walls of the rectangle.
        # List of intersection points. Should have 2 points. AKA: These are the 2 points where the line intersects the
        # walls of the rectangle.
        ips_lst = []

        x_TOP = (rect_bounds[3] - b) / m
        if x_TOP >= rect_bounds[0] and x_TOP <= rect_bounds[2]:
            ips_lst.append((x_TOP, rect_bounds[3]))

        x_BOTTOM = (rect_bounds[1] - b) / m
        if x_BOTTOM >= rect_bounds[0] and x_BOTTOM <= rect_bounds[2]:
            ips_lst.append((x_BOTTOM, rect_bounds[1]))

        y_RIGHT = b + (m*rect_bounds[2])
        if y_RIGHT >= rect_bounds[1] and y_RIGHT <= rect_bounds[3]:
            ips_lst.append((rect_bounds[2], y_RIGHT))

        y_LEFT = b + (m*rect_bounds[0])
        if y_LEFT >= rect_bounds[1] and y_LEFT <= rect_bounds[3]:
            ips_lst.append((rect_bounds[0], y_LEFT))

        return LineString([ips_lst[0], ips_lst[1]])

def find_intersection(line_string, rails_gdf, show_map=False):
    """
    Finds the first intersection point between a given line and a set of railroad segments.

    Inputs:
    - line_string (LineString): The input line to check for intersections.
    - rails_gdf (GeoDataFrame): GeoDataFrame of railroad line segments (LineStrings).
    - show_map (bool, optional): If True, displays a map for debugging.

    Returns:
    - intersection_points (Point or int): Intersection point if found, else -1.
    - rails_inds_intersecting (int): Row index of the intersected rail segment, or -1 if none.
    - int: Total number of intersections found.
    """
    # Check to see which railroad segments intersect with the straight-line path from the station
    # to the emeregency location.
    intersections = rails_gdf.geometry.intersection(line_string)

    def get_first_point(geom):
        if geom.is_empty:
            return None
        elif isinstance(geom, Point):
            return geom
        elif isinstance(geom, MultiPoint):
            return list(geom.geoms)[0]  # or geom.geoms[0]
        else:
            return None

    # a GeoSeries of "LineString Z EMPTY" shapes + one or more intersection Points + possibly other types of intersection Shapely shapes
    filtered_intersects = intersections[intersections.apply(lambda x: not (x.is_empty))]
    filtered_intersects = filtered_intersects.apply(get_first_point)

    if len(filtered_intersects) == 0:
        # None of the lines in "rails_gdf" intersected with the given line.
        if show_map:
            fig, ax = plt.subplots(figsize = (20,20))
            rails_gdf.plot(ax=ax, color='darkgreen')
            gpd.GeoDataFrame(geometry=[line_string], crs="EPSG:4326").plot(ax=ax, color='black')
            plt.show()
            print("END OF find_intersection")
            print("-1, -1,", filtered_intersects.size)
        return None, None, 0

    # One of the intersections between the straight line (from the station to the emergency) and the railroads.
    intersection_points = filtered_intersects.tolist()

    # The index of one of the rail line segments that intersected with the straight line.
    rails_inds_intersecting = filtered_intersects.index.tolist()

    # plot if we set to boolean input to true
    if show_map:
        fig, ax = plt.subplots(figsize = (20,20))
        rails_gdf.plot(ax=ax, color='darkgreen')
        gpd.GeoDataFrame(geometry=[line_string], crs="EPSG:4326").plot(ax=ax, color='black')
        gpd.GeoDataFrame(geometry=filtered_intersects, crs="EPSG:4326").plot(ax=ax, color='purple')
        crossings_gdf.plot(ax=ax, color='orange')
        plt.show()
    return intersection_points, rails_inds_intersecting, len(intersection_points)

def show_map_of_buffered_line(buffer_polygon, num_crossings_within, point1, point2, intersection_point, line, google_api_path):
    """
    Plots two maps: one showing the buffered area and crossings, and a second close-up around the intersection point.

    Inputs:
    - buffer_polygon (Polygon): Buffered area around the line.
    - num_crossings_within (int): Number of crossings found within the buffered area.
    - point1 (Point): Starting point of the line.
    - point2 (Point): Ending point of the line.
    - intersection_point (Point): Point where the line intersects a railroad.
    - line (LineString): The original extended line segment.
    - google_api_path (LineString): Line obtained from Google Maps API for visualization.

    Returns none.
    """

    # plot 1 (buffered area and crossings)
    fig, ax = plt.subplots(figsize = (20,20))
    gpd.GeoSeries([buffer_polygon]).plot(ax=ax, color='cornflowerblue')
    rails_gdf_within_East_districts.plot(ax=ax, color='darkgreen')
    crossings_gdf.plot(ax=ax, color='gray', markersize=10)
    gpd.GeoSeries([point1]).plot(ax=ax, color='black')
    gpd.GeoSeries([point2]).plot(ax=ax, color='purple')
    gpd.GeoSeries([LineString([point1, point2])]).plot(ax=ax, color='red', linestyle='dashed')
    gpd.GeoSeries([google_api_path]).plot(ax=ax, color='black')
    gpd.GeoSeries([intersection_point]).plot(ax=ax, color='red')
    ax.set_title("MAP OF BUFFERED POLYGON num_crossings_within=" + str(num_crossings_within) + " LAST CROSSING (BLACK) AND EMERGENCY (PURPLE)")
    plt.show()

    # plot 2 (close-up)
    radius_angle = 0.00274
    x_min, x_max = intersection_point.x - radius_angle, intersection_point.x + radius_angle # set limits
    y_min, y_max = intersection_point.y - radius_angle, intersection_point.y + radius_angle
    fig, ax = plt.subplots(figsize = (20,20))
    gpd.GeoSeries([buffer_polygon]).plot(ax=ax, color='cornflowerblue')
    rails_gdf_within_East_districts.plot(ax=ax, color='green')
    gpd.GeoSeries([line]).plot(ax=ax, color='darkgreen')
    crossings_gdf.plot(ax=ax, color='orange')
    gpd.GeoSeries([intersection_point]).plot(ax=ax, color='red')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("CLOSE-UP MAP AROUND INTERSECTION POINT num_crossings_within=" + str(num_crossings_within) + " LAST CROSSING (BLACK) AND EMERGENCY (PURPLE)")
    plt.show()

def get_crossings_list(intersection_point, rails_ind_intersecting, point1, point2, buffer_width=304.8, show_map=False, google_api_path=None):
    """
    Returns a GeoDataFrame of crossings near an extended railroad line segment.

    Inputs:
    - intersection_point (Point): A point on the railroad line.
    - rails_ind_intersecting (int): Index of the rail segment in 'rails_gdf_within_East_districts'.
    - point1, points (Point): Starting point and ending point of the emergency route.
    - buffer_width (float, optional): Width (in meters) of the search buffer around the rail line.
    - show_map (bool, optional): If true, plots maps for visualization.
    - google_api_path (LineString, optional): Path from Google API for reference on the map.

    Returns:
    - GeoDataFrame: Crossings within the buffer zone around the extended railroad line.
    """
    # Find the nearest crossings on that rail-line
    line = rails_gdf_within_East_districts.loc[rails_ind_intersecting, 'geometry']
    m, b = None, None

    # Calculate the slope and y-intercept of that rail line segment.
    if line.coords[1][0] != line.coords[0][0]:
        m = (line.coords[1][1] - line.coords[0][1]) / (line.coords[1][0] - line.coords[0][0])
        b = line.coords[0][1] - m*line.coords[0][0]
    
    # Extend the rail line so that it fills all of the space of the whole map of East Houston (viewing_bounds).
    extended_line = extend_line_to_length(m, b, line.coords[0][0], intersection_point, rect_bounds=None, standard_length=3500)
    
    # Convert the extended version of the rail line from its original CRS 4326 to the projected CRS 3857
    line_projected = LineString([transformer.transform(x, y) for x, y in extended_line.coords])
    
    # Buffer the line by 1000 feet (converted to meters: 1000 ft is about 304.8 meters)
    buffer_polygon = line_projected.buffer(buffer_width)
    
    # Create GeoDataFrame and set the CRS to EPSG:3857, because buffer_polygon is in EPSG:3857.
    gdf_3857 = gpd.GeoDataFrame(geometry=[buffer_polygon], crs="EPSG:3857")
    
    # Transform the GeoDataFrame to EPSG:4326
    parallelogram_4326 = gdf_3857.to_crs("EPSG:4326")

    # Get all the crossings within this buffer zone of the extended rail line.
    crossings_within_poly = crossings_gdf[crossings_gdf.geometry.within(parallelogram_4326['geometry'].iloc[0])]

    crossings_within_poly = crossings_within_poly[['CrossingID', 'geometry']]

    if show_map is True:
        print("type(point1)=", type(point1), "type(point2)=", type(point2))
        show_map_of_buffered_line(parallelogram_4326['geometry'].iloc[0], len(crossings_within_poly), point1, point2, intersection_point, line, google_api_path)
    
    return crossings_within_poly

def get_blockage_bool(crossing_ind):
    """
    Checks if a given crossing is blocked at any point based on crossing predictions.
    Input:
    - crossing_ind (int): Index of the crossing in 'crossing_preds'.

    Returns 1 if the crossing is blocked at any time, 0 otherwise.
    """
    has_ones = (crossing_preds.loc[crossing_ind] == 1).any()
    return 1 if has_ones is True else 0


def compute_time_from_station(distance_info, distance_info_ind, cur_station_ind, cur_station, emergency, crossings_gdf,
                              ignore_other_districts=False, max_crossings_allowed=1,
                              crossing_preds_entries_units="seconds", crossing_preds_intervs_units="minutes", show_maps=False):
    """
    Computes travel time from a station to an emergency site using Mapbox Directions API,
    optionally accounting for railroad crossings. Optional inputs not explained for brevity.

    Inputs:
    - distance_info (dict): Dictionary to update with time and crossing information.
    - distance_info_ind (int): Index for the current station entry.
    - cur_station_ind (int): Index of the station.
    - cur_station (Point): Station location (EPSG:4326).
    - emergency (Point): Emergency site location (EPSG:4326).
    - crossings_gdf (GeoDataFrame): Railroad crossings dataset.

    Returns none.
    """

    global NUM_STATIONS_MORE_THAN_ALLOWED_NUM_CROSSINGS
    global NUM_STATIONS_IN_DIFF_DISTRICT
    global NUM_STATIONS_CROSSINGS_NOT_WITHIN_BUFFER

    # initialize distance info fields
    distance_info["time"] = None
    distance_info["crossing_ids"] = []
    distance_info["crossings_blocked"] = []
    distance_info["station_ind"] = cur_station_ind

    # Project current station to 3395 for distance measurements
    cur_station_3395 = Point(transformer_4326_to_3395.transform(cur_station.x, cur_station.y))
    dist_ind_of_station = get_district_ind_of_point(cur_station)
    if (ignore_other_districts) and (dist_ind_of_emer != dist_ind_of_station):
        distance_info["time"] = UNREACHABLE_TIME
        NUM_STATIONS_IN_DIFF_DISTRICT += 1
        return

    # set up Mapbox API request
    env = Env()
    env.read_env(".env")
    mapbox_token = env.str("MAPBOX_ACCESS_TOKEN")
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving/"
        f"{cur_station.x},{cur_station.y};{emergency.x},{emergency.y}"
        f"?access_token={mapbox_token}&overview=full&geometries=polyline")

    # attempt to fetch route from Mapbox API
    encoded_polyline = None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        route = data["routes"][0]
        distance_info["time"] = route["legs"][0]["duration"] / 60.0  # seconds to minutes
        encoded_polyline = route["geometry"]
    except Exception as e:
        distance_info["time"] = UNREACHABLE_TIME
        return
    
    # Find intersections between route and railroads
    path_as_linestring = LineString([(lon, lat) for lat, lon in decode_polyline(encoded_polyline)])
    intersection_points, rails_inds_intersecting, num_intersections = find_intersection(
        path_as_linestring, rails_gdf_within_East_districts, show_map=show_maps)

    if intersection_points is not None:
        # Check if too many crossings
        if num_intersections > max_crossings_allowed:
            distance_info["sl_dist"] = -1
            NUM_STATIONS_MORE_THAN_ALLOWED_NUM_CROSSINGS += 1
            return
        distance_info["would_need_crossing"] = True
        last_crossing = cur_station

        # go through each crossing to find and sort the viable ones
        for i in range(num_intersections):
            viable_crossings = get_crossings_list(
                intersection_points[i], rails_inds_intersecting[i], last_crossing, emergency, show_map=False)

            if len(viable_crossings) < 1:
                distance_info["sl_dist"] = -1
                distance_info["unable_to_find_crossing"] = True
                NUM_STATIONS_CROSSINGS_NOT_WITHIN_BUFFER += 1
                return

            distance_info["unable_to_find_crossing"] = False

            # Sort viable crossings by proximity to intersection
            viable_crossings_3395 = viable_crossings.to_crs(epsg=3395)
            viable_crossings_3395['distance'] = viable_crossings_3395.geometry.distance(intersection_points[i])
            viable_crossings_3395 = viable_crossings_3395.sort_values(by='distance', ascending=True)

            # choose closest crossing
            closest_crossing_ind = viable_crossings_3395['distance'].idxmin()
            chosen_crossing = viable_crossings_3395.loc[closest_crossing_ind]

            # update distance_info with crossing ID and blockage status
            distance_info["crossing_ids"].append(chosen_crossing['CrossingID'])
            is_blocked = get_blockage_bool(chosen_crossing.name)
            distance_info["crossings_blocked"].append(is_blocked)

            # Update last crossing to the chosen one
            last_crossing = chosen_crossing['geometry']
            last_crossing = Point(reverse_transformer_3395_to_4326.transform(last_crossing.x, last_crossing.y))
    
    else:  # no intersection points
        distance_info["would_need_crossing"] = False

    # set time as unreacheable if we couldn't get there
    if distance_info["time"] is None:
        distance_info["time"] = UNREACHABLE_TIME


def compute_time(check_crossing, emergency_obj, stations, ignore_other_districts=False, max_crossings_allowed=10, 
                 crossing_preds_entries_units="seconds", crossing_preds_intervs_units="minutes", show_maps=False, output_type="common"):
    """
    Computes travel times from stations to an emergency site, considering potential railroad blockages.

    Inputs:
    - check_crossing (function): Function that predicts crossing blockages by time.
    - emergency_obj (pd.Series): Emergency information with 'emergency_type', 'time', and 'coordinates'.
    - stations (dict[int, tuple[float, float]]): Station IDs mapped to their coordinates (latitude, longitude).

    - ignore_other_districts (bool, optional): If True, exclude stations outside the emergency district.
    - max_crossings_allowed (int, optional): Maximum crossings allowed between station and emergency.
    - crossing_preds_entries_units (str, optional): Unit for blockage prediction entries (defaults to "seconds").
    - crossing_preds_intervs_units (str, optional): Unit for blockage prediction intervals (defaults to "minutes").
    - show_maps (bool, optional): If True, plots maps for debugging purposes.
    - output_type (str, optional): "common" for dict outputs, otherwise returns a DataFrame.

    Returns:
    - If output_type == "common":
        - dict: Mapping of station IDs to (time, noisy_time) tuples.
        - dict: Mapping of station IDs to (crossing IDs, blockage flags).
    - Else:
        - pd.DataFrame: DataFrame summarizing distances and travel times per station.
    """

    global leave_time, distance_info_columns, crossing_preds, dist_ind_of_emer, crossings_gdf
    global UNREACHABLE_TIME, NUM_STATIONS_MORE_THAN_ALLOWED_NUM_CROSSINGS, NUM_STATIONS_CROSSINGS_NOT_WITHIN_BUFFER
    global NUM_STATIONS_IN_DIFF_DISTRICT
    UNREACHABLE_TIME = 1440
    NUM_STATIONS_MORE_THAN_ALLOWED_NUM_CROSSINGS = 0
    NUM_STATIONS_IN_DIFF_DISTRICT = 0
    NUM_STATIONS_CROSSINGS_NOT_WITHIN_BUFFER = 0

    # a GeoDataFrame of locations of railroad crossings in Houston that we allow this function to consider.
    crossings_gdf = gpd.read_file(TOP_FOLDER + "/" + SHAPEFILES_FOLDER + "/blockages_gdf_avgd_East")
    cur_time = emergency_obj["time"].time()
    leave_time = emergency_obj["time"]
    date = emergency_obj["time"].date()
    emergency_loc = Point(emergency_obj["coordinates"][::-1])

    # Set the local variable blockage_preds
    # blockage_preds.loc[Crossing i][Index of time period j] is 1 or 0,
    # depending on if the crossing is expected to be blocked or not at that time period.
    blockage_preds = check_crossing(cur_time, date)
    crossing_preds = blockage_preds
    
    # If ignore_other_districts is True, then this is used in the for loop to see if the stations are in the same district as the emergency site.
    dist_ind_of_emer = get_district_ind_of_point(emergency_loc)

    # Notes on the column meanings:
    #     "sl_" prefix means straight-line or a sum of two straight-line distances.
    #     "road_dist" means the distance that Google Maps calculates.
    #     "would_need_crossing" means that the route would need a crossing (even if the function
    #     can not find a crossing).
    distance_info_columns = ["station_ind", "crossing_ids", "crossings_blocked", "sl_dist", "road_dist", "time", "would_need_crossing", 
                             "unable_to_find_crossing", "crossing_ind", "crossing_id", "sl_dist_to_crossing", "sl_dist_crossing_to_emer", 
                             "road_dist_to_crossing", "road_dist_crossing_to_emer", "time_to_crossing", "time_crossing_to_emer", "found_exact_time_interv", "time_blockage"]

    distance_info_ind, distance_info = 0, {}
    for station_ind in stations.keys():
        distance_info[station_ind] = {}

    global emergency_loc_3395
    emergency_loc_3395 = Point(transformer_4326_to_3395.transform(emergency_loc.x, emergency_loc.y))

    # Go through all of the stations and compute the time from station to emergency
    for cur_station_ind, cur_station_loc in stations.items():
        # You have to pass in Longitude, Latitude to construct a point.
        cur_station = Point(cur_station_loc[1], cur_station_loc[0])

        # TODO: will try to reduce number of inputs . . .
        compute_time_from_station(distance_info[cur_station_ind], distance_info_ind, cur_station_ind, cur_station, emergency_loc, 
                                  crossings_gdf, ignore_other_districts=ignore_other_districts, max_crossings_allowed=max_crossings_allowed, 
                                  crossing_preds_entries_units=crossing_preds_entries_units, crossing_preds_intervs_units=crossing_preds_intervs_units, show_maps=show_maps)
        distance_info_ind += 1

    # return of this function changes depending on input
    if output_type == "common":
        distance_info_time = {station: (station_val["time"], 0 if station_val["time"] == UNREACHABLE_TIME else add_noise(station_val["time"]) ) for station, station_val in distance_info.items()}
        distance_info_crossings = {station: (station_val["crossing_ids"], station_val["crossings_blocked"]) for station, station_val in distance_info.items()}
        return distance_info_time, distance_info_crossings
    else: # used for zach's testing
        df = pd.DataFrame(distance_info)
        return df
