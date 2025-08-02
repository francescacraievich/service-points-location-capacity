# Service Points Location and Capacity Problem

Implementation of the mixed-integer linear programming (MILP) model from:
> Raviv, T. (2023). "The service points' location and capacity problem". 
> Transportation Research Part E, 176, 103216.

## Overview
This project implements an optimization model for locating and sizing automatic 
parcel lockers (APLs) in urban delivery networks, considering stochastic demand 
and pickup processes.

## 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run basic tests
Run the main test file to validate the model with a small instance:
```bash
python test.py
```
### 3. Run scalability tests (optional)
For comprehensive scalability analysis with real Austrian city data:
```bash
python scalability.py
```
service-points-location-capacity/
├── data/                               # Data files and outputs
│   └── output/                         # Results directory
│       ├── osm/                        # OpenStreetMap data
│       │   └── austria-latest.osm.pbf  # OSM data for Austrian cities
│       ├── scalability/                # Scalability test results
│       └── test/                       # Basic test outputs
│           ├── instance.json           # Saved test instance
│           ├── solution.json           # Solution details
│           ├── network.png             # Network visualization
│           ├── figure2_rejection_function.png
│           └── figure4_model_comparison.png
├── src/                                # Source code
│   ├── models/                         # Optimization models
│   │   ├── __init__.py                 # Package initializer
│   │   ├── deterministic_model.py      # Deterministic benchmark model
│   │   ├── rejection_function.py       # Rejection function calculations
│   │   ├── scenarios_model.py          # Scenario-based stochastic model
│   │   └── sp_model.py                 # Main PWL stochastic model (Raviv 2023)
│   └── utils/                          # Utility functions
│       ├── __init__.py                 # Package initializer
│       ├── data_generator.py           # Synthetic instance generator
│       ├── osm_austria.py              # Real data loader from OpenStreetMap
│       └── visualization.py            # Plotting and visualization tools
├── .gitignore                          # Git ignore file
├── config.json                         # Configuration parameters (optional)
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── scalability.py                      # Scalability tests with real Austrian data
└── test.py                             # Main test file and entry point


## Key Files Description

### **test.py**
Main entry point for testing the Service Point Location and Capacity model. Includes:
- Basic model validation
- Comparison between PWL, deterministic, and scenario-based models
- Small instance tests (4×4 grid)

### **scalability.py**
Comprehensive scalability analysis using real data from Austrian cities:
- Tests with Vienna, Graz, and Linz data from OpenStreetMap
- Computational complexity analysis
- Performance benchmarking for different instance sizes
- Generates detailed reports and visualizations

###  **src/models/**
Core optimization models implementing Raviv (2023):
- `sp_model.py`: Main stochastic model with piecewise linear approximation
- `deterministic_model.py`: Benchmark deterministic model
- `scenarios_model.py`: Two-stage stochastic model with demand scenarios
- `rejection_function.py`: M/M/k queueing-based rejection calculations

###  **src/utils/**
Supporting utilities:
- `data_generator.py`: Create synthetic and realistic test instances
- `osm_austria.py`: Extract and process real city data from OpenStreetMap
- `visualization.py`: Generate network plots and analysis charts

###  **data/output/**
Results organized by test type:
- `test/`: Basic model validation results
- `scalability/`: Performance analysis with timestamped runs

- `osm/`: OpenStreetMap data files for Austrian cities
