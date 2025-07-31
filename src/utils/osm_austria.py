"""
Minimal OSM processor for Austria data following Raviv (2023)
"""

import osmium
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class AustriaOSMHandler(osmium.SimpleHandler):
    """Simple handler to extract data from Austria OSM file"""
    
    # City bounds from Raviv (2023)
    CITIES = {
        'Vienna': {
            'bounds': (16.18, 48.11, 16.58, 48.32),
            'target_demand_points': 17701,
            'target_sp_candidates': 4260
        },
        'Graz': {
            'bounds': (15.34, 46.99, 15.51, 47.15),
            'target_demand_points': 6777,
            'target_sp_candidates': 1459
        },
        'Linz': {
            'bounds': (14.22, 48.24, 14.35, 48.35),
            'target_demand_points': 3351,
            'target_sp_candidates': 658
        }
    }
    
    def __init__(self, city: str):
        super().__init__()
        self.city = city
        self.bounds = self.CITIES[city]['bounds']
        self.buildings = []
        self.parking = []
        
    def node(self, n):
        """Extract parking and transport nodes"""
        if not self._in_bounds(n.location.lon, n.location.lat):
            return
            
        if ('amenity' in n.tags and n.tags['amenity'] == 'parking') or \
           ('railway' in n.tags and n.tags['railway'] in ['station', 'halt']):
            self.parking.append((n.location.lon, n.location.lat))
    
    def way(self, w):
        """Extract buildings"""
        if 'building' in w.tags and len(w.nodes) > 0:
            # Simple check if in bounds (would need full geometry ideally)
            self.buildings.append(w.id)
    
    def _in_bounds(self, lon, lat):
        min_lon, min_lat, max_lon, max_lat = self.bounds
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def load_austria_instance(city: str = "Vienna", osm_file: str = None) -> Dict:
    """
    Load city data from austria-latest.osm.pbf
    
    Parameters:
    -----------
    city : str
        City name (Vienna, Graz, or Linz)
    osm_file : str, optional
        Path to the OSM file. If None, tries default location
    """
    import os
    
    if osm_file is None:
        osm_file = "data/osm/austria-latest.osm.pbf"
    
    if not os.path.exists(osm_file):
        raise FileNotFoundError(
            f"OSM file not found at: {osm_file}\n"
            f"Full path: {os.path.abspath(osm_file)}"
        )
    
    print(f"Loading {city} from OSM data: {osm_file}")
    
    # IMPORTANTE: Crea l'handler PRIMA di usarlo!
    handler = AustriaOSMHandler(city)
    
    # This processes the entire file - may take a minute
    handler.apply_file(osm_file, locations=True)
    
    # Create 250m grid as in paper
    bounds = handler.bounds
    cell_size = 0.0025  # ~250m in degrees
    
    # Generate demand points on grid
    demand_points = []
    demand_rates = []
    
    lon = bounds[0]
    while lon < bounds[2]:
        lat = bounds[1]
        while lat < bounds[3]:
            # Center of grid cell
            x = (lon + cell_size/2 - (bounds[0]+bounds[2])/2) * 111000
            y = (lat + cell_size/2 - (bounds[1]+bounds[3])/2) * 111000
            demand_points.append((x, y))
            demand_rates.append(np.random.uniform(0.5, 10))  # Random demand
            lat += cell_size
        lon += cell_size
    
    # Convert parking locations to meters
    sp_locations = []
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    for lon, lat in handler.parking[:handler.CITIES[city]['target_sp_candidates']]:
        x = (lon - center_lon) * 111000
        y = (lat - center_lat) * 111000
        sp_locations.append((x, y))
    
    # Trim to match paper's numbers
    n_demand = min(len(demand_points), handler.CITIES[city]['target_demand_points'])
    n_sp = min(len(sp_locations), handler.CITIES[city]['target_sp_candidates'])
    
    # Se non ci sono abbastanza SP locations, generane di casuali
    if len(sp_locations) < n_sp:
        print(f"Warning: Only found {len(sp_locations)} parking locations, generating random SP locations")
        while len(sp_locations) < n_sp:
            x = np.random.uniform(-10000, 10000)
            y = np.random.uniform(-10000, 10000)
            sp_locations.append((x, y))
    
    instance = {
        "city": city,
        "demand_points": demand_points[:n_demand],
        "demand_rates": demand_rates[:n_demand],
        "sp_locations": sp_locations[:n_sp],
        "service_radius": 300,  # 300m for real instances
        "n_demand_points": n_demand,
        "n_sp_locations": n_sp,
        "total_demand": sum(demand_rates[:n_demand])
    }
    
    print(f"Loaded {city}: {n_demand} demand points, {len(sp_locations[:n_sp])} SP candidates")
    return instance