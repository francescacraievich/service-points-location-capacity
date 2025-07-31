"""
Data generation utilities for creating synthetic instances
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional


def generate_synthetic_instance(
        grid_size: int = 10,
        service_radius: float = 600,
        seed: Optional[int] = None,
        demand_range: Tuple[float, float] = (0.5, 10.0)
    ) -> Dict:
    """
    Generate a synthetic instance for the SP location problem
    
    Parameters:
    -----------
    grid_size : int
        Size of the grid (grid_size x grid_size)
    service_radius : float
        Service radius in meters
    seed : int, optional
        Random seed for reproducibility
    demand_range : tuple
        Min and max demand rates
        
    Returns:
    --------
    dict
        Instance data with demand points, SP locations, and parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Grid spacing (200m as in the paper)
    spacing = 200
    
    # Generate demand points (center of each cell)
    demand_points = []
    demand_rates = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * spacing + spacing / 2
            y = j * spacing + spacing / 2
            demand_points.append((x, y))
            
            # Random demand rate
            rate = np.random.uniform(demand_range[0], demand_range[1])
            demand_rates.append(rate)
    
    # Generate candidate SP locations (even rows and columns)
    sp_locations = []
    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            x = i * spacing + spacing / 2
            y = j * spacing + spacing / 2
            sp_locations.append((x, y))
    
    # Calculate total area
    area_km2 = (grid_size * spacing / 1000) ** 2
    
    instance = {
        "grid_size": grid_size,
        "spacing": spacing,
        "area_km2": area_km2,
        "demand_points": demand_points,
        "demand_rates": demand_rates,
        "sp_locations": sp_locations,
        "service_radius": service_radius,
        "total_demand": sum(demand_rates),
        "n_demand_points": len(demand_points),
        "n_sp_locations": len(sp_locations)
    }
    
    print(f"Generated instance:")
    print(f"  Area: {area_km2:.2f} km²")
    print(f"  Demand points: {len(demand_points)}")
    print(f"  SP candidate locations: {len(sp_locations)}")
    print(f"  Total demand: {sum(demand_rates):.2f}")
    
    return instance


def generate_realistic_instance(
        city_size: str = "small",
        seed: Optional[int] = None
    ) -> Dict:
    """
    Generate a more realistic instance with non-uniform demand
    
    Parameters:
    -----------
    city_size : str
        Size category: "small", "medium", "large"
    seed : int, optional
        Random seed
        
    Returns:
    --------
    dict
        Instance data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # City size parameters
    sizes = {
        "small": {"grid": 10, "centers": 2, "radius": 600},
        "medium": {"grid": 20, "centers": 4, "radius": 600},
        "large": {"grid": 30, "centers": 6, "radius": 800}
    }
    
    params = sizes[city_size]
    spacing = 200
    
    # Generate demand points with clustering around centers
    demand_points = []
    demand_rates = []
    
    # Random city centers (high demand areas)
    centers = []
    for _ in range(params["centers"]):
        cx = np.random.randint(2, params["grid"]-2) * spacing
        cy = np.random.randint(2, params["grid"]-2) * spacing
        centers.append((cx, cy))
    
    # Generate demand points
    for i in range(params["grid"]):
        for j in range(params["grid"]):
            x = i * spacing + spacing / 2
            y = j * spacing + spacing / 2
            demand_points.append((x, y))
            
            # Base demand
            base_demand = np.random.uniform(0.5, 2.0)
            
            # Add center influence
            for cx, cy in centers:
                dist_to_center = np.sqrt((x - cx)**2 + (y - cy)**2)
                influence = max(0, 1 - dist_to_center / (params["grid"] * spacing / 3))
                base_demand += influence * np.random.uniform(5, 15)
            
            demand_rates.append(base_demand)
    
    # Generate SP locations (more dense near centers)
    sp_locations = []
    
    # Regular grid
    for i in range(0, params["grid"], 3):
        for j in range(0, params["grid"], 3):
            x = i * spacing + spacing / 2
            y = j * spacing + spacing / 2
            sp_locations.append((x, y))
    
    # Additional locations near centers
    for cx, cy in centers:
        for _ in range(5):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(200, 800)
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            # Check bounds
            if 0 <= x <= params["grid"] * spacing and 0 <= y <= params["grid"] * spacing:
                sp_locations.append((x, y))
    
    # Remove duplicates
    sp_locations = list(set(sp_locations))
    
    return {
        "city_size": city_size,
        "grid_size": params["grid"],
        "spacing": spacing,
        "area_km2": (params["grid"] * spacing / 1000) ** 2,
        "demand_points": demand_points,
        "demand_rates": demand_rates,
        "sp_locations": sp_locations,
        "service_radius": params["radius"],
        "city_centers": centers,
        "total_demand": sum(demand_rates),
        "n_demand_points": len(demand_points),
        "n_sp_locations": len(sp_locations)
    }


def save_instance(instance: Dict, filename: str):
    """Save instance to JSON file"""
    with open(filename, 'w') as f:
        json.dump(instance, f, indent=2)
    print(f"Instance saved to {filename}")


def load_instance_from_file(filename: str) -> Dict:
    """Load instance from JSON file"""
    with open(filename, 'r') as f:
        instance = json.load(f)
    print(f"Instance loaded from {filename}")
    return instance


def generate_case_study_instance(city: str = "Vienna") -> Dict:
    """
    Generate instance based on case study cities from the paper
    
    Parameters:
    -----------
    city : str
        City name: "Linz", "Graz", or "Vienna"
        
    Returns:
    --------
    dict
        Instance approximating the city
    """
    # Approximate data from Table 1 in the paper
    cities = {
        "Linz": {
            "population": 206724,
            "n_demand_points": 3351,
            "n_sp_candidates": 658,
            "area_km2": 96
        },
        "Graz": {
            "population": 291245,
            "n_demand_points": 6777,
            "n_sp_candidates": 1459,
            "area_km2": 128
        },
        "Vienna": {
            "population": 1911274,
            "n_demand_points": 17701,
            "n_sp_candidates": 4260,
            "area_km2": 415
        }
    }
    
    if city not in cities:
        raise ValueError(f"Unknown city: {city}")
    
    data = cities[city]
    
    # Approximate grid size
    grid_size = int(np.sqrt(data["n_demand_points"]))
    spacing = int(np.sqrt(data["area_km2"] * 1e6 / data["n_demand_points"]))
    
    print(f"Generating {city} instance (approximation)...")
    print(f"  Grid: {grid_size}x{grid_size}, spacing: {spacing}m")
    
    # Generate points
    demand_points = []
    demand_rates = []
    
    # Simplified generation
    for i in range(grid_size):
        for j in range(grid_size):
            if len(demand_points) >= data["n_demand_points"]:
                break
            x = i * spacing
            y = j * spacing
            demand_points.append((x, y))
            
            # Demand proportional to population
            avg_demand = data["population"] / data["n_demand_points"] * 0.02
            demand_rates.append(np.random.uniform(0.5 * avg_demand, 1.5 * avg_demand))
    
    # SP locations (simplified)
    sp_locations = []
    sp_spacing = int(np.sqrt(data["area_km2"] * 1e6 / data["n_sp_candidates"]))
    
    for i in range(0, grid_size * spacing, sp_spacing):
        for j in range(0, grid_size * spacing, sp_spacing):
            if len(sp_locations) >= data["n_sp_candidates"]:
                break
            sp_locations.append((i, j))
    
    return {
        "city": city,
        "population": data["population"],
        "area_km2": data["area_km2"],
        "demand_points": demand_points[:data["n_demand_points"]],
        "demand_rates": demand_rates[:data["n_demand_points"]],
        "sp_locations": sp_locations[:data["n_sp_candidates"]],
        "service_radius": 300,  # As in the paper
        "total_demand": sum(demand_rates[:data["n_demand_points"]]),
        "n_demand_points": len(demand_points[:data["n_demand_points"]]),
        "n_sp_locations": len(sp_locations[:data["n_sp_candidates"]])
    }
def load_osm_instance(osm_file: str, city: str = "Vienna") -> Dict:
    """
    Load instance from OSM data following Raviv (2023) methodology
    
    Parameters:
    -----------
    osm_file : str
        Path to .osm.pbf file
    city : str
        City name (Vienna, Graz, or Linz)
        
    Returns:
    --------
    dict
        Instance data
    """
    from .osm_processor import extract_city_data, aggregate_to_grid, CITY_BOUNDS
    
    if city not in CITY_BOUNDS:
        raise ValueError(f"Unknown city: {city}")
    
    print(f"Extracting {city} data from OSM...")
    
    # Extract raw data
    city_data = extract_city_data(osm_file, city, CITY_BOUNDS[city])
    
    print(f"Found {len(city_data['demand_points'])} demand sources")
    print(f"Found {len(city_data['sp_candidates'])} SP candidates")
    
    # Aggregate to grid as in paper
    demand_points = aggregate_to_grid(
        city_data['demand_points'], 
        cell_size=250,  # 250m cells as in paper
        bounds=CITY_BOUNDS[city]
    )
    
    # Convert coordinates to meters (local projection)
    # For simplicity, using approximate conversion
    center_lat = (CITY_BOUNDS[city][1] + CITY_BOUNDS[city][3]) / 2
    meters_per_degree = 111000 * np.cos(np.radians(center_lat))
    
    # Convert to local coordinates in meters
    center_lon = (CITY_BOUNDS[city][0] + CITY_BOUNDS[city][2]) / 2
    
    demand_points_meters = []
    for p in demand_points:
        x = (p['x'] - center_lon) * meters_per_degree
        y = (p['y'] - center_lat) * 111000
        demand_points_meters.append((x, y))
    
    sp_locations_meters = []
    for sp in city_data['sp_candidates']:
        x = (sp['lon'] - center_lon) * meters_per_degree
        y = (sp['lat'] - center_lat) * 111000
        sp_locations_meters.append((x, y))
    
    # Create instance
    instance = {
        "city": city,
        "source": "OpenStreetMap",
        "demand_points": demand_points_meters,
        "demand_rates": [p['demand_rate'] for p in demand_points],
        "sp_locations": sp_locations_meters,
        "service_radius": 300,  # 300m as in paper for real instances
        "total_demand": sum(p['demand_rate'] for p in demand_points),
        "n_demand_points": len(demand_points_meters),
        "n_sp_locations": len(sp_locations_meters),
        "metadata": {
            "cell_size": 250,
            "bounds": CITY_BOUNDS[city],
            "extraction_date": datetime.now().isoformat()
        }
    }
    
    print(f"\nProcessed {city} instance:")
    print(f"  - {len(demand_points_meters)} demand cells")
    print(f"  - {len(sp_locations_meters)} SP candidates")
    print(f"  - Total demand: {instance['total_demand']:.1f}")
    
    return instance