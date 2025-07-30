# Service Points Location and Capacity Problem

Implementation of the mixed-integer linear programming (MILP) model from:
> Raviv, T. (2023). "The service points' location and capacity problem". 
> Transportation Research Part E, 176, 103216.

## Overview
This project implements an optimization model for locating and sizing automatic 
parcel lockers (APLs) in urban delivery networks, considering stochastic demand 
and pickup processes.

## Installation
```bash
pip install -r requirements.txt

```
Complete structure of repository:

service-points-location-capacity/
│
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── input/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sp_model.py
│   │   └── rejection_function.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   └── visualization.py
│   └── main.py
├── tests/
│   └── test_model.py
└── notebooks/
    └── analysis.ipynb