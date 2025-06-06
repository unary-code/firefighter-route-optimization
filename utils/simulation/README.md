# Simulation Scripts

This README provides instructions on how to run the `run_simulation.py` script located in this folder and a brief explanation of its use.

## Running the Simulation

To run the `run_simulation.py` test script, execute the following command:

```sh
python -m utils.simulation.run_simulation
```

## Purpose of the Script

The `run_simulation.py` script is designed to simulate various scenarios for the HFD train dispatch system. It helps in analyzing and optimizing the dispatch process.
The script opens pre-ran simulations from the `results` folder, prints results and shows a density plot comparing method a (dispatch geographically closest) and
method b (taking blockages into account).

## Dependencies

The `run_simulation.py` script primarily relies on a simulator object created `simulation.py` which processes assignments through a MIP (abstracted in `dispatch.py`)
and simulates real emergencies when given a date. It also relies heavily on `compute_time.py`, where we actually calculate the time it would take for a vehicle to reach
the emergency.

