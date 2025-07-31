"""
Implementation of the rejection function based on DTMC analysis
As described in Section 3.1 of Raviv (2023)
"""

import numpy as np
from typing import Dict, Tuple


class RejectionFunction:
    """Calculate expected rejections using DTMC approach"""
    
    def __init__(self, capacity: int, pickup_prob: float = 0.5):
        """
        Initialize rejection function
        
        Parameters:
        -----------
        capacity : int
            Number of lockers in the service point
        pickup_prob : float
            Probability of pickup in each period (p)
        """
        self.C = capacity
        self.p = pickup_prob
        
    def expected_rejections(self, lambda_rate: float) -> float:
        """
        Calculate expected number of rejections per period
        Using Erlang B formula for M/M/C/C queue
        """
        if lambda_rate == 0:
            return 0.0
            
        # For very large capacities, use fluid approximation
        if self.C >= 150:
            effective_capacity = self.C * self.p
            return max(0, lambda_rate - effective_capacity)
        
        # For smaller capacities, use Erlang B formula
        rho = lambda_rate / self.p  # offered load
        
        # Calculate Erlang B blocking probability
        blocking_prob = self._erlang_b(rho, self.C)
        
        # Expected rejections = arrival rate * blocking probability
        return lambda_rate * blocking_prob
    
    def _erlang_b(self, offered_load: float, servers: int) -> float:
        """
        Calculate Erlang B blocking probability
        P_block = (A^C / C!) / sum(A^k / k! for k=0 to C)
        where A = offered load
        """
        if offered_load == 0:
            return 0.0
            
        # Use iterative calculation to avoid overflow
        inv_b = 1.0
        for k in range(1, servers + 1):
            inv_b = 1.0 + inv_b * k / offered_load
            
        return 1.0 / inv_b
    
    def get_piecewise_linear_points(self, max_lambda: float, n_points: int = 12) -> Dict[str, list]:
        """
        Get points for piecewise linear approximation
        Focus more points around the critical region (rho ≈ 1)
        """
        # Create non-uniform spacing with more points around rho = 1
        rho_values = []
        
        # Points from 0 to 0.8 (sparse)
        rho_values.extend(np.linspace(0, 0.8, int(n_points * 0.3)))
        
        # Points from 0.8 to 1.2 (dense)
        rho_values.extend(np.linspace(0.8, 1.2, int(n_points * 0.5)))
        
        # Points from 1.2 to max (sparse)
        max_rho = max_lambda / (self.C * self.p)
        if max_rho > 1.2:
            rho_values.extend(np.linspace(1.2, max_rho, int(n_points * 0.2)))
        
        # Convert to lambda values
        lambda_values = [rho * self.C * self.p for rho in rho_values]
        lambda_values = [lam for lam in lambda_values if lam <= max_lambda]
        
        # Ensure we include 0 and max_lambda
        if lambda_values[0] > 0:
            lambda_values.insert(0, 0)
        if lambda_values[-1] < max_lambda:
            lambda_values.append(max_lambda)
        
        rejection_values = [self.expected_rejections(lam) for lam in lambda_values]
        
        return {
            'lambda_values': lambda_values,
            'rejection_values': rejection_values
        }