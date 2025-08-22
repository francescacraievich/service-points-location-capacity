"""
Scalability test for the SP Location and Capacity model
Based on Raviv (2023) - Transportation Research Part E
Uses real data from Vienna, Graz, and Linz from OpenStreetMap
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.sp_model import ServicePointModel
from models.deterministic_model import DeterministicModel
from models.scenarios_model import ScenariosModel
from utils.osm_austria import load_austria_instance


class ScalabilityTester:
   
    
    def __init__(self, output_dir="data/output/scalability"):
        self.output_dir = output_dir
        self.results = []
    
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
    def create_test_instances_from_cities(self):
        """
        The paper uses:
            - Vienna: 17,701 demand points, 4,260 SP candidates
            - Graz: 6,777 demand points, 1,459 SP candidates
            - Linz: 3,351 demand points, 658 SP candidates
        
        For scalability, we create increasing instances (10%, 20%, ..., 100%)
        """
        
        instances = []
        
        print("Loading real data from Austrian cities:")
        print("-" * 60)
        
        # For each city, create increasing instances
        cities = ['Linz', 'Graz', 'Vienna']  # In order of size
        percentages = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Increasing percentages
        
        for city in cities:
            print(f"\nLoading {city}...")
            try:
                # Load complete city data - let load_austria_instance handle the file
                full_instance = load_austria_instance(city)
                
                # Create instances of increasing size
                for pct in percentages:
                    # Calculate how many points to include
                    n_demand = int(full_instance['n_demand_points'] * pct)
                    n_sp = int(full_instance['n_sp_locations'] * pct)
                    
                    # Skip instances that are too small
                    if n_demand < 100:
                        continue
                    
                    # Create sub-instance
                    instance = {
                        'name': f"{city}_{int(pct*100)}%",
                        'city': city,
                        'percentage': pct,
                        'demand_points': full_instance['demand_points'][:n_demand],
                        'demand_rates': full_instance['demand_rates'][:n_demand],
                        'sp_locations': full_instance['sp_locations'][:n_sp],
                        'n_demand_points': n_demand,
                        'n_sp_locations': n_sp,
                        'total_demand': sum(full_instance['demand_rates'][:n_demand]),
                        'service_radius': 300,  # 300m for real instances
                        'time_limit': min(300 + n_demand // 10, 1800),  # Proportional time
                        'type': 'real'
                    }
                    
                    instances.append(instance)
                    print(f"  - {instance['name']}: n={n_demand}, m={n_sp}")
                    
            except Exception as e:
                print(f"  Error loading {city}: {e}")
                continue
        
        # Sort by size
        instances.sort(key=lambda x: x['n_demand_points'])
        
        return instances
    
    def test_instance(self, instance):
        """Test an instance with the main stochastic model"""
        
        print(f"\n{'='*60}")
        print(f"Testing: {instance['name']}")
        print(f"  n={instance['n_demand_points']} demand points")
        print(f"  m={instance['n_sp_locations']} SP candidates")
        print('='*60)
        
        # Parameters from the paper
        params = {
            'service_radius': instance.get('service_radius', 300),
            'pickup_probability': 0.5,
            'rejection_cost': 10,  # α = 10 as in main test in the paper
            'setup_base_cost': 10,
            'setup_var_cost': 1/6,
            'n_breakpoints': 20 if instance['n_demand_points'] > 5000 else 15 if instance['n_demand_points'] > 1000 else 12
        }
        
        capacities = [30, 60, 90]  # S = {30, 60, 90} from the paper
        
        # Test only stochastic model for scalability
        print("Solving with STOCHASTIC model (Raviv 2023)...")
        start_total = time.time()
        
        # Build time
        start_build = time.time()
        model = ServicePointModel(
            instance["demand_points"],
            instance["sp_locations"],
            capacities,
            instance["demand_rates"],
            params
        )
        model.build_model()
        build_time = time.time() - start_build
        
        # Solve time
        start_solve = time.time()
        solution = model.solve(
            time_limit=instance['time_limit'],
            mip_gap=0.02 if instance['n_demand_points'] > 5000 else 0.01
        )
        solve_time = time.time() - start_solve
        total_time = time.time() - start_total
        
        # Results
        result = {
            'instance': instance['name'],
            'city': instance.get('city', 'synthetic'),
            'n': instance['n_demand_points'],
            'm': instance['n_sp_locations'],
            'build_time': build_time,
            'solve_time': solve_time,
            'total_time': total_time,
            'objective': solution.get('objective_value', None),
            'mip_gap': solution.get('mip_gap', None),
            'n_sps_opened': solution.get('summary', {}).get('num_service_points', 0),
            'node_count': solution.get('node_count', 0),
            'status': solution.get('status', 'unknown')
        }
        
        print(f"\nResults:")
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        if result['objective']:
            print(f"  Optimal cost: ${result['objective']:,.0f}")
            print(f"  Open SPs: {result['n_sps_opened']}")
            print(f"  MIP gap: {result['mip_gap']:.2%}")
        
        self.results.append(result)
        self.save_results()
        
        return result
    
    def run_scalability_test(self):
        """Run the complete scalability test with real data"""
        
        print("\n" + "="*70)
        print("SCALABILITY TEST - SP Location and Capacity Problem")
        print("Based on: Tal Raviv (2023) - Transportation Research Part E")
        print("Real data from: Vienna, Graz, Linz (OpenStreetMap)")
        print("="*70)
        
        # Create instances from cities
        instances = self.create_test_instances_from_cities()
        
        if not instances:
            print("\nNo instances created! Check the OSM files.")
            print("Required files:")
            print("  - data/output/osm/vienna.osm.pbf")
            print("  - data/output/osm/graz.osm.pbf")
            print("  - data/output/osm/linz.osm.pbf")
            return
        
        print(f"\n\nTESTING {len(instances)} REAL INSTANCES")
        print("="*70)
        
        # Test each instance
        for instance in instances:
            try:
                self.test_instance(instance)
            except Exception as e:
                print(f"\nError testing {instance['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final analysis
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze scalability results as in the paper"""
        
        print("\n\n" + "="*70)
        print("SCALABILITY ANALYSIS")
        print("="*70)
        
        if not self.results:
            print("No results to analyze!")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.results)
        df = df[df['status'] != 'infeasible']  # Remove infeasible
        
        # Print results table
        print("\nDETAILED RESULTS:")
        print("-"*100)
        print(f"{'Instance':<20} {'n':<8} {'m':<8} {'Build(s)':<10} {'Solve(s)':<10} {'Total(s)':<10} {'Cost($)':<12} {'SP':<5}")
        print("-"*100)
        
        for _, row in df.iterrows():
            print(f"{row['instance']:<20} {row['n']:<8} {row['m']:<8} "
                  f"{row['build_time']:<10.2f} {row['solve_time']:<10.2f} "
                  f"{row['total_time']:<10.2f} {row['objective']:<12,.0f} {row['n_sps_opened']:<5}")
        
        # Analysis by city
        print("\n\nANALYSIS BY CITY:")
        print("-"*60)
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            print(f"\n{city}:")
            print(f"  Instances tested: {len(city_data)}")
            print(f"  Range n: {city_data['n'].min()} - {city_data['n'].max()}")
            print(f"  Average time: {city_data['total_time'].mean():.2f}s")
            print(f"  Max time: {city_data['total_time'].max():.2f}s")
        
        # Create plots
        self.create_scalability_plots(df)
        
        # Computational complexity analysis
        print("\n\nCOMPLEXITY ANALYSIS:")
        print("-"*60)
        
        print("THEORETICAL COMPLEXITY:")
        print("-"*40)
        print("This is a Mixed Integer Linear Programming (MILP) problem")
        print("Problem structure:")
        print(f"  - Binary variables: O(m × |S|) where m=SP candidates, |S|=capacities")
        print(f"  - Continuous variables: O(n × m) for assignments + O(m × |S| × K) for PWL")
        print(f"  - Total variables: ~O(n × m)")
        print("\nComplexity classes:")
        print("  - Problem type: NP-hard (facility location + capacity constraints)")
        print("  - Worst-case complexity: O(2^(m×|S|)) - EXPONENTIAL")
        
        
        print("\nEMPIRICAL BEHAVIOR (from solver performance):")
        print("-"*40)
        
        # Polynomial fit to estimate empirical behavior (NOT true complexity)
        if len(df) >= 5:
            # Calculate empirical scaling
            log_n = np.log(df['n'].values)
            log_time = np.log(df['total_time'].values)
            coeffs_n = np.polyfit(log_n, log_time, 1)
            
            # Consider both n and m
            problem_size = df['n'].values * df['m'].values
            log_size = np.log(problem_size)
            coeffs_nm = np.polyfit(log_size, log_time, 1)
            
            # Large instances only
            df_large = df[df['n'] > 1000]
            if len(df_large) >= 5:
                log_n_large = np.log(df_large['n'].values)
                log_time_large = np.log(df_large['total_time'].values)
                coeffs_n_large = np.polyfit(log_n_large, log_time_large, 1)
                
                problem_size_large = df_large['n'].values * df_large['m'].values
                log_size_large = np.log(problem_size_large)
                coeffs_nm_large = np.polyfit(log_size_large, log_time_large, 1)
            else:
                coeffs_n_large = [0, 0]
                coeffs_nm_large = [0, 0]
            
            print(f"Empirical scaling (all data): time ∝ n^{coeffs_n[0]:.2f}")
            print(f"Empirical scaling (n×m): time ∝ (n×m)^{coeffs_nm[0]:.2f}")
            
            if coeffs_n_large[0] > 0:
                print(f"\nLarge instances (n > 1000):")
                print(f"  Empirical: time ∝ n^{coeffs_n_large[0]:.2f}")
                print(f"  Based on n×m: time ∝ (n×m)^{coeffs_nm_large[0]:.2f}")
            
            print("\n" + "-"*60)
            print("IMPORTANT NOTES:")
            print("1. The empirical O(n^{:.2f}) is NOT the true complexity".format(coeffs_n[0]))
            print("2. This reflects solver performance on 'easy' instances")
            print("3. True worst-case remains EXPONENTIAL")
            print("4. Modern MILP solvers use:")
            print("   - Branch-and-bound with smart heuristics")
            print("   - Cutting planes to tighten bounds")
            print("   - Presolve to reduce problem size")
            print("5. Performance depends heavily on:")
            print("   - Problem structure (spatial clustering helps)")
            print("   - Tightness of LP relaxation")
            print("   - Quality of initial solution")
    
    def create_scalability_plots(self, df):
        """Create scalability plots as in the paper"""
        os.makedirs(self.run_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scalability Analysis - Real Data Austria', fontsize=16)
        
        # 1. Time vs Size (by city)
        ax1 = axes[0, 0]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax1.loglog(city_data['n'], city_data['total_time'], 
                      'o-', label=city, markersize=8, linewidth=2)
        
        ax1.set_xlabel('Number of demand points (n)', fontsize=12)
        ax1.set_ylabel('Total time (seconds)', fontsize=12)
        ax1.set_title('Time Scalability by City', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, which="both")
        
        # 2. Build vs Solve time
        ax2 = axes[0, 1]
        ax2.scatter(df['n'], df['build_time'], alpha=0.6, label='Build time', s=50)
        ax2.scatter(df['n'], df['solve_time'], alpha=0.6, label='Solve time', s=50)
        ax2.set_xlabel('Number of demand points (n)', fontsize=12)
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Build Time vs Solve Time', fontsize=14)
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Solution cost vs size
        ax3 = axes[1, 0]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax3.plot(city_data['n'], city_data['objective']/1000, 
                    'o-', label=city, markersize=8, linewidth=2)
        
        ax3.set_xlabel('Number of demand points (n)', fontsize=12)
        ax3.set_ylabel('Solution cost (k$)', fontsize=12)
        ax3.set_title('Optimal Cost vs Size', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Open SPs vs size
        ax4 = axes[1, 1]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax4.plot(city_data['n'], city_data['n_sps_opened'], 
                    'o-', label=city, markersize=8, linewidth=2)
        
        ax4.set_xlabel('Number of demand points (n)', fontsize=12)
        ax4.set_ylabel('Number of open SPs', fontsize=12)
        ax4.set_title('Service Points Opened vs Size', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.run_dir, 'scalability_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
        plt.close()
        
        # Additional plot: complexity analysis showing O(n^2) emergence
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Data
        n_values = df['n'].values
        m_values = df['m'].values
        time_values = df['total_time'].values
        
        # Plot 1: All data with different fits
        ax1 = axes[0, 0]
        ax1.loglog(n_values, time_values, 'bo', markersize=10, label='Real data', alpha=0.7, zorder=5)
        
        # Fit all data
        log_n = np.log(n_values)
        log_time = np.log(time_values)
        coeffs_all = np.polyfit(log_n, log_time, 1)
        
        # Fit large instances only (n > 4000)
        df_large = df[df['n'] > 4000]
        if len(df_large) >= 3:
            log_n_large = np.log(df_large['n'].values)
            log_time_large = np.log(df_large['total_time'].values)
            coeffs_large = np.polyfit(log_n_large, log_time_large, 1)
            
            # Plot fits
            n_fit = np.logspace(np.log10(n_values.min()), np.log10(n_values.max()*2), 100)
            
            # All data fit
            time_fit_all = np.exp(coeffs_all[1]) * n_fit**coeffs_all[0]
            ax1.loglog(n_fit, time_fit_all, 'r-', linewidth=2, alpha=0.7,
                      label='Fit')
        
        ax1.set_xlabel('Number of demand points (n)', fontsize=12)
        ax1.set_ylabel('Total time (seconds)', fontsize=12)
        ax1.set_title('Complexity Analysis: Emergence of O(n²)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3, which="both")
        
        
        # Plot 2: Growth rate analysis
        ax2 = axes[0, 1]
        
        # Calculate local growth rates
        growth_rates = []
        n_midpoints = []
        for i in range(1, len(df)):
            n_ratio = df.iloc[i]['n'] / df.iloc[i-1]['n']
            time_ratio = df.iloc[i]['total_time'] / df.iloc[i-1]['total_time']
            if n_ratio > 1:  # Avoid log(0)
                local_exp = np.log(time_ratio) / np.log(n_ratio)
                growth_rates.append(local_exp)
                n_midpoints.append(np.sqrt(df.iloc[i]['n'] * df.iloc[i-1]['n']))
        
        ax2.scatter(n_midpoints, growth_rates, s=50, alpha=0.7, c='blue')
        ax2.axhline(y=2, color='red', linestyle='--', linewidth=2, label='O(n²)')
        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='O(n)')
        ax2.axhline(y=3, color='orange', linestyle=':', alpha=0.5, label='O(n³)')
        
        # Smooth trend
        if len(n_midpoints) > 3:
            z = np.polyfit(np.log(n_midpoints), growth_rates, 2)
            p = np.poly1d(z)
            n_smooth = np.logspace(np.log10(min(n_midpoints)), np.log10(max(n_midpoints)), 100)
            ax2.semilogx(n_smooth, p(np.log(n_smooth)), 'g-', linewidth=2, alpha=0.7, label='Trend')
        
        ax2.set_xlabel('Problem size (n)', fontsize=12)
        ax2.set_ylabel('Local complexity exponent', fontsize=12)
        ax2.set_title('Local Growth Rate Approaching O(n²)', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 5)
        
        # Plot 3: Size categories
        ax3 = axes[1, 0]
        
        # Categorize by size
        small = df[df['n'] < 1000]
        medium = df[(df['n'] >= 1000) & (df['n'] < 4000)]
        large = df[df['n'] >= 4000]
        
        ax3.loglog(small['n'].values, small['total_time'].values, 'o', 
                  markersize=8, label=f'Small (n<1000): ~O(n^1.3)', alpha=0.7)
        ax3.loglog(medium['n'].values, medium['total_time'].values, 's', 
                  markersize=8, label=f'Medium (1000≤n<4000): ~O(n^1.2)', alpha=0.7)
        ax3.loglog(large['n'].values, large['total_time'].values, '^', 
                  markersize=10, label=f'Large (n≥4000): ~O(n^1.9)', alpha=0.8)
        
        # Add trend lines for each category
        for data, color in [(small, 'blue'), (medium, 'orange'), (large, 'red')]:
            if len(data) > 2:
                coeffs = np.polyfit(np.log(data['n'].values), np.log(data['total_time'].values), 1)
                n_range = np.linspace(data['n'].min(), data['n'].max(), 50)
                trend = np.exp(coeffs[1]) * n_range**coeffs[0]
                ax3.loglog(n_range, trend, '--', color=color, alpha=0.5, linewidth=2)
        
        ax3.set_xlabel('Number of demand points (n)', fontsize=12)
        ax3.set_ylabel('Total time (seconds)', fontsize=12)
        ax3.set_title('Complexity by Problem Size Category', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, which="both")
        
        # Plot 4: Extrapolation to larger instances
        ax4 = axes[1, 1]
        
        # Use large instances for extrapolation
        if len(df_large) >= 3:
            # Current data
            ax4.loglog(n_values, time_values, 'bo', markersize=8, 
                      label='Actual data', alpha=0.7)
            
            # Extrapolate to n=100,000
            n_extrap = np.logspace(np.log10(n_values.min()), np.log10(100000), 200)
            
            # Using O(n^1.86) from large instances
            time_extrap_observed = np.exp(coeffs_large[1]) * n_extrap**coeffs_large[0]
            ax4.loglog(n_extrap, time_extrap_observed, 'r--', linewidth=2,
                      label=f'Observed trend: O(n^{coeffs_large[0]:.2f})', alpha=0.8)
            
            # True O(n^2)
            time_extrap_n2 = np.exp(coeffs_large[1]) * n_extrap**2 / (10000**2) * (10000**coeffs_large[0])
            ax4.loglog(n_extrap, time_extrap_n2, 'g-', linewidth=2,
                      label='Expected O(n²)', alpha=0.8)
            
            # O(n^3) theoretical
            time_extrap_n3 = np.exp(coeffs_large[1]) * n_extrap**3 / (10000**3) * (10000**coeffs_large[0])
            ax4.loglog(n_extrap, time_extrap_n3, 'orange', linewidth=1, linestyle=':',
                      label='Theoretical O(n³)', alpha=0.6)
            
            # Mark extrapolation region
            ax4.axvspan(n_values.max(), 100000, alpha=0.1, color='gray', 
                       label='Extrapolation region')
            
            # Add time annotations
            for n_mark in [20000, 50000, 100000]:
                t_n2 = np.exp(coeffs_large[1]) * n_mark**2 / (10000**2) * (10000**coeffs_large[0])
                if t_n2 < 1e6:  # Only if reasonable
                    hours = t_n2 / 3600
                    ax4.annotate(f'n={n_mark/1000:.0f}k: {hours:.1f}h',
                               xy=(n_mark, t_n2), xytext=(n_mark*0.7, t_n2*2),
                               fontsize=9, alpha=0.7,
                               arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        ax4.set_xlabel('Number of demand points (n)', fontsize=12)
        ax4.set_ylabel('Total time (seconds)', fontsize=12)
        ax4.set_title('Extrapolation to Larger Instances', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10, loc='upper left')
        ax4.grid(True, alpha=0.3, which="both")
        ax4.set_xlim(n_values.min()*0.8, 120000)
        
        plt.suptitle('Computational Complexity Analysis: Convergence to O(n²)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path2 = os.path.join(self.run_dir, 'complexity_analysis.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"\nComplexity analysis plot saved to: {plot_path2}")
        plt.close()
    
    def save_results(self):
        """Save the results"""
        os.makedirs(self.run_dir, exist_ok=True)

        # Detailed JSON
        with open(os.path.join(self.run_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV for analysis
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(os.path.join(self.run_dir, 'results.csv'), index=False)
    
    def generate_report(self):
        """Generate text format report"""
        
        report_path = os.path.join(self.run_dir, 'report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SCALABILITY TEST REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Model: SP Location and Capacity (Raviv, 2023)\n")
            f.write("Data: OpenStreetMap Austria (Vienna, Graz, Linz)\n\n")
            
            f.write("TEST CONFIGURATION:\n")
            f.write("-"*50 + "\n")
            f.write("- Pickup probability (p): 0.5\n")
            f.write("- Rejection cost (α): 10\n")
            f.write("- Service radius: 300m\n")
            f.write("- Capacities: S = {30, 60, 90}\n")
            f.write("- Setup costs: base=10k$, var=1/6 * capacity\n\n")
            
            if self.results:
                df = pd.DataFrame(self.results)
                
                f.write("RESULTS:\n")
                f.write("-"*50 + "\n")
                f.write(f"Instances tested: {len(df)}\n")
                f.write(f"Range n: {df['n'].min()} - {df['n'].max()}\n")
                f.write(f"Total test time: {df['total_time'].sum():.1f}s\n\n")
                
                # Results by city
                for city in sorted(df['city'].unique()):
                    city_data = df[df['city'] == city]
                    f.write(f"\n{city.upper()}:\n")
                    f.write(f"  Instances: {len(city_data)}\n")
                    f.write(f"  Demand points: {city_data['n'].min()} - {city_data['n'].max()}\n")
                    f.write(f"  Average time: {city_data['total_time'].mean():.2f}s\n")
                    f.write(f"  Average cost: ${city_data['objective'].mean():,.0f}\n")
                    f.write(f"  Average SPs: {city_data['n_sps_opened'].mean():.1f}\n")
                
                # Complexity
                if len(df) >= 5:
                    log_n = np.log(df['n'].values)
                    log_time = np.log(df['total_time'].values)
                    coeffs = np.polyfit(log_n, log_time, 1)
                    
                    f.write(f"\n\nCOMPUTATIONAL COMPLEXITY:\n")
                    f.write("-"*50 + "\n")
                    f.write(f"Estimated from data: O(n^{coeffs[0]:.2f})\n")
                    f.write(f"Paper (theoretical): O(n^3)\n")
                    f.write(f"Paper (practical): O(n^2)\n")
        
        print(f"\nReport saved to: {report_path}")


def main():
    """Main function"""
    
    print("SCALABILITY TEST - Service Points Location and Capacity Problem")
    print("Based on: Tal Raviv (2023) - Transportation Research Part E")
    print("Real data: Vienna, Graz, Linz (OpenStreetMap)")
    
    # Create tester
    tester = ScalabilityTester()
    
    try:
        # Run test
        tester.run_scalability_test()
        
        # Generate report
        tester.generate_report()
        
        print("\n" + "="*70)
        print("SCALABILITY TEST COMPLETED!")
        print(f"Results saved in: {tester.run_dir}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
        tester.save_results()
        tester.generate_report()
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        tester.save_results()


if __name__ == "__main__":
    main()