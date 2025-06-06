# HFD Train Dispatch - Spring 2025

## Project Overview
This repository contains code and data for the HFD Train Dispatch system, aimed at optimizing train scheduling and dispatch policies. The project includes raw data processing, data exploration, predictive modeling, simulation, and visualization tools.

Please visit exploration.demo.ipynb to visualize our data exploration results. Also visit simulation_demo.ipynb to run our simulation and see some 
preloaded results (as the simulation may take hours to run).

---

## Repository Structure

```
HFD_train_dispatch_Sp25/
│── raw_data/
│   ├── TRAINFO/
│   ├── ... (Other raw data files)
│
│── clean_data/
│   ├── all_shapefiles_in_east_houston (for mapping purposes)
│   ├── blockage_maps (scratchwork maps of blockages)
|
│   ├── blockage_data.csv files (many different matrices of processed data)
│   ├── ... (Other raw data files)

│-- .env (where we keep our MAPBOX API key)
│── exploration/
│   ├── blockage_data_cleaning.py
│   ├── dispatch_data_cleaning.py
│   ├── blockage_data_exploration.py

│── prediction_model/
│   ├── prediction.py
│   ├── ml_model.py
│   ├── ... (This module is obsolete.)

│── results/
│   ├── .pkl files where we stored various runs of our simulation

│── utils/
│   ├── dispatch/
│   │   ├── dispatch.py
│   ├── simulation/
│   │   ├── simulation.py
│   │   ├── compute_time.py
│   │   ├── compute_time_basic.py (naive version of method B)
│   │   ├── run_simulation.py
│   ├── visualization/
│   │   ├── visualization.py

│── simulation_demo.ipynb 
│── exploration_demo.ipynb
│── requirements.txt
```
---

## Data Pipeline

1. **Raw Data (`raw_data/`)**
   - Contains the original datasets, including train information and other relevant data.
   
2. **Cleaned Data (`clean_data/`)**
   - Processed datasets ready for analysis and modeling.

3. **Data Cleaning and Exploration (`exploration/`)**
   - Running the data cleaning file will generate files in the `clean_data/` folder.
   - Running the exploration file will output visualizations and statistics about the data.

4. *Prediction Machine Learning Modeling
  - **Prediction (`prediction_model/`)**
  - Includes machine learning models for predicting train blockages.
  - *Note: This is obsolete as of March 13, 2025.*

5. **Utils (`utils/`)**
   - Scripts for simulation, modeling, prediction, and visualization
   - **Simulation (`simulation/`)**
     - Scripts for simulating train dispatch scenarios.
   - **Dispatch Policy (`dispatch/`)**
     - Contains scripts for defining and testing train dispatch policies.
   - **Visualization (`visualization/`)**
     - Tools for visualizing dispatch policies and results.

6. **Demos (`simulation_demo.ipynb & exploration_demo.ipynb`)**
   - These are demo notebooks for our project. Run the cells to see some results
   of data exploration and simulation.
   - *Note: The simulation may take a few hours to run. You can run a cell immediately to see our pre-saved results. 
   Also, the pre-saved results are from 5 days, one from each of Jan - May 2019 (our poster results were separated day-by-day -- these are aggregated). *

---

## Setup Instructions

### Requirements
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```
Note: You will need a Gurobi license to properly use the gurobipy library. You can get an academic 
license here: https://www.gurobi.com/academia/academic-program-and-licenses/?gad_source=1&gclid=Cj0KCQjwhYS_BhD2ARIsAJTMMQaqj0irWwlt_hZyY_-RyExXGOqvb12FoLMQIUiFPZy45iAachkkiPkaAk_yEALw_wcB.

Note: To calculate distances, we used an API by MapBox. The key to use this api is in the .env in the outermost directory. Since MapBox is free, this key may expire if we 
run out of API calls. To read more about getting an API key, go to this link: https://www.mapbox.com/.

### Running Data Exploration
To generate some basic results in your terminal run the following command:
```bash
python -m exploration.blockage_data_exploration
```
To run some of the plots of the data exploration go to demo.ipynb and run the cells.
More instructions and information can be found in the notebook.

### Running Data Cleaning
To generate cleaned data, navigate to the `exploration/` folder and run the following:
```bash
python -m exploration.blockage_data_cleaning
python -m exploration.dispatch_data_cleaning
```
You will see the the first few rows of some cleaned data output for a sanity check.

### Running Simulation
To simulate dispatch scenarios, run:
```bash
python -m utils.simulation.run_simulation
```
For more information, please see the README.md file in the simulation subfolder.

### Running Predictions
This is an obsolete part of our project that attempted to predict future blockages.
If you want to optionally take a look, run the terminal command below.

To run the basic predictions script that predicts if there's blockage on Jan 2025 from 7-8AM on
crossing 288224V, you can run:
```bash
python -m prediction_model.prediction
```
Please be aware that this is a preliminary LSTM model that needs tweaks to change and may not
be entirely accurate.

### Running Visualization
To visualize the dispatch map, run this script:
```bash
python -m utils.visualization.visualization
```

---

## Contribution Guidelines
1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push and create a pull request:
   ```bash
   git push origin feature-name
   ```

---


