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
├── test.py              # File principale per test semplice
├── scalability.py       # File principale per test scalabilità
├── requirements.txt
├── README.md
├── config.json
├── src/                 # Codice sorgente
│   ├── models/
│   └── utils/
└── data/               # Dati e output