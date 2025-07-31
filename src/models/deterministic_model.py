"""
Deterministic model variant for comparison (Figure 4)
Uses average demand without stochasticity
"""

import gurobipy as gp
from gurobipy import GRB
from .sp_model import ServicePointModel
import numpy as np


class DeterministicModel(ServicePointModel):
    """
    Deterministic variant with relaxed capacity constraints
    Uses safety factor instead of hard constraints
    """
    
    def __init__(self, *args, safety_factor: float = 0.8, **kwargs):
        """
        Parameters:
        -----------
        safety_factor : float
            Factor to reduce effective capacity (default 0.8 = use 80% of capacity)
        """
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor
        self.model_type = "deterministic"
    
    def _prepare_pwl_data(self):
        """No PWL approximation needed for deterministic model"""
        pass
    
    def build_model(self):
        """Build deterministic MILP model"""
        print(f"Building DETERMINISTIC model (safety factor={self.safety_factor})...")
        
        # Index sets
        D_idx = range(len(self.D))
        F_idx = range(len(self.F))
        S_idx = range(len(self.C))
        
        # Decision variables
        print("Creating decision variables...")
        
        # y[f,s] = 1 if SP with capacity s opened at location f
        self.y = self.model.addVars(
            F_idx, S_idx, 
            vtype=GRB.BINARY, 
            name="y"
        )
        
        # x[d,f] = flow from demand point d to SP f
        self.x = self.model.addVars(
            [(d, f) for d in D_idx for f in F_idx 
             if self.distances[(d, f)] <= self.r],
            lb=0, 
            vtype=GRB.CONTINUOUS, 
            name="x"
        )
        
        # Objective: Only setup costs (no rejection costs in deterministic)
        print("Setting objective (setup costs only)...")
        obj = gp.quicksum(
            self._get_setup_cost(f, s) * self.y[f, s]
            for f in F_idx for s in S_idx
        )
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        print("Adding constraints...")
        
        # At most one capacity per location
        for f in F_idx:
            self.model.addConstr(
                gp.quicksum(self.y[f, s] for s in S_idx) <= 1,
                name=f"one_size_per_location_{f}"
            )
        
        # Satisfy all demand (deterministic assumes all demand must be served)
        for d in D_idx:
            eligible_sps = [f for f in F_idx if self.distances[(d, f)] <= self.r]
            if eligible_sps:  # Only add constraint if there are eligible SPs
                self.model.addConstr(
                    gp.quicksum(self.x[d, f] for f in eligible_sps) == self.mu[d],
                    name=f"satisfy_demand_{d}"
                )
        
        # RELAXED CAPACITY CONSTRAINT with safety factor
        # This makes the model more conservative but feasible
        for f in F_idx:
            for s in S_idx:
                # Use only a fraction of the capacity (safety factor)
                safe_capacity = self.C[s] * self.p * self.safety_factor
                
                self.model.addConstr(
                    gp.quicksum(
                        self.x[d, f] 
                        for d in D_idx 
                        if (d, f) in self.x
                    ) <= safe_capacity * self.y[f, s],
                    name=f"safe_capacity_{f}_{s}"
                )
        
        # Closest SP constraint (same as stochastic)
        for d in D_idx:
            for f in F_idx:
                if self.distances[(d, f)] <= self.r:
                    farther_sps = [
                        fp for fp in F_idx 
                        if self.distances[(d, fp)] > self.distances[(d, f)]
                        and self.distances[(d, fp)] <= self.r
                    ]
                    
                    if farther_sps:
                        self.model.addConstr(
                            gp.quicksum(self.x[d, fp] for fp in farther_sps) <= 
                            self.mu[d] * (1 - gp.quicksum(self.y[f, s] for s in S_idx)),
                            name=f"closest_sp_{d}_{f}"
                        )
        
        self.model.update()
        
        print(f"\nDeterministic model statistics:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        print(f"  Safety factor: {self.safety_factor}")
    
    def solve(self, **kwargs):
        """Override solve to handle deterministic specifics"""
        solution = super().solve(**kwargs)
        
        # Add model type to solution
        solution['model_type'] = 'deterministic'
        solution['safety_factor'] = self.safety_factor
        
        return solution