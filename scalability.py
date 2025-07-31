"""
Test di scalabilità per il modello SP Location and Capacity
Basato su Raviv (2023) - Transportation Research Part E
Include test con istanze sintetiche e dati reali di Vienna, Graz, Linz
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
from utils.data_generator import generate_synthetic_instance


class ScalabilityTester:
    """
    Test di scalabilità seguendo la metodologia del paper
    """
    
    def __init__(self, output_dir="data/output/scalability"):
        self.output_dir = output_dir
        self.results = []
        
        # Crea directory output con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
    def create_test_instances(self):
        """
        Crea istanze di test seguendo il paper
        Sezione 5.2 - Computational experiments
        """
        
        # Configurazioni sintetiche dal paper
        test_configs = [
            # (name, grid_size, sp_density, time_limit)
            ("XS", 5, 0.5, 60),      # Extra small: 25 points
            ("S", 8, 0.4, 120),      # Small: 64 points  
            ("M", 11, 0.3, 300),     # Medium: 121 points (come paper)
            ("L", 15, 0.25, 600),    # Large: 225 points
            ("XL", 20, 0.2, 1200),   # Extra large: 400 points
            ("XXL", 25, 0.15, 1800), # XXL: 625 points
        ]
        
        instances = []
        
        print("Generazione istanze di test (seguendo struttura paper):")
        print("-" * 60)
        
        for name, grid_size, sp_density, time_limit in test_configs:
            print(f"\n{name} ({grid_size}x{grid_size} grid)...")
            
            # Genera istanza sintetica
            instance = generate_synthetic_instance(
                grid_size=grid_size,
                service_radius=600,  # r = 600m per istanze sintetiche
                seed=42,
                demand_range=(3.0, 7.0)  # Domanda uniforme intorno a 5
            )
            
            # Riduci SP candidates in base alla densità
            n_sp_target = int(instance['n_demand_points'] * sp_density)
            if len(instance['sp_locations']) > n_sp_target:
                instance['sp_locations'] = instance['sp_locations'][:n_sp_target]
                instance['n_sp_locations'] = n_sp_target
            
            instance['name'] = name
            instance['time_limit'] = time_limit
            instance['type'] = 'synthetic'
            instances.append(instance)
            
            print(f"  - Punti domanda (n): {instance['n_demand_points']}")
            print(f"  - Candidati SP (m): {instance['n_sp_locations']}")
            print(f"  - Domanda totale: {instance['total_demand']:.1f}")
            print(f"  - Densità SP: {instance['n_sp_locations']/instance['n_demand_points']:.2f}")
        
        return instances
    
    def test_instance(self, instance, capacities=None):
        """Testa una singola istanza seguendo il setup del paper"""
        
        if capacities is None:
            # Capacità dal paper
            if instance.get('type') == 'real':
                capacities = [30, 60, 90, 120, 150]  # Per città reali
            else:
                capacities = [30, 60, 90]  # Per istanze sintetiche
        
        print(f"\n{'='*60}")
        print(f"Testing: {instance['name']}")
        print(f"  n={instance['n_demand_points']} demand points")
        print(f"  m={instance['n_sp_locations']} SP candidates")
        print(f"  |S|={len(capacities)} capacity options")
        print('='*60)
        
        # Parametri dal paper
        params = {
            'service_radius': instance.get('service_radius', 600),
            'pickup_probability': 0.5,
            'rejection_cost': 10,
            'setup_base_cost': 10,
            'setup_var_cost': 1/6,
            'n_breakpoints': 12 if instance['n_demand_points'] < 100 else 8
        }
        
        # Record timing
        start_total = time.time()
        
        try:
            # Crea modello
            print("\nCostruendo modello MILP...")
            model = ServicePointModel(
                demand_points=instance["demand_points"],
                candidate_locations=instance["sp_locations"],
                capacities=capacities,
                demand_rates=instance["demand_rates"],
                params=params
            )
            
            # Build model
            start_build = time.time()
            model.build_model()
            build_time = time.time() - start_build
            
            print(f"  Tempo costruzione: {build_time:.2f}s")
            print(f"  Variabili: {model.model.NumVars}")
            print(f"  Vincoli: {model.model.NumConstrs}")
            print(f"  Non-zeri: {model.model.NumNZs}")
            
            # Solve
            print(f"\nRisolvendo (time limit: {instance['time_limit']}s)...")
            solution = model.solve(
                time_limit=instance['time_limit'],
                mip_gap=0.02 if instance['n_demand_points'] > 200 else 0.01
            )
            
            total_time = time.time() - start_total
            
            # Raccogli risultati
            result = {
                'name': instance['name'],
                'type': instance.get('type', 'synthetic'),
                'n': instance['n_demand_points'],
                'm': instance['n_sp_locations'],
                'total_demand': instance['total_demand'],
                'n_variables': model.model.NumVars,
                'n_constraints': model.model.NumConstrs,
                'n_nonzeros': model.model.NumNZs,
                'build_time': build_time,
                'solve_time': solution['runtime'],
                'total_time': total_time,
                'status': 'optimal' if solution['status'] == 2 else 'feasible',
                'objective': solution.get('objective_value', None),
                'gap': solution.get('mip_gap', None),
                'nodes': solution.get('node_count', 0),
                'time_limit': instance['time_limit']
            }
            
            if solution.get('summary'):
                result.update({
                    'n_sps_opened': solution['summary']['num_service_points'],
                    'total_capacity': solution['summary']['total_capacity'],
                    'avg_utilization': solution['summary']['avg_utilization'],
                    'setup_cost': solution['summary']['total_setup_cost'],
                    'rejection_cost': solution['summary']['total_rejection_cost'],
                    'total_rejections': solution['summary']['expected_total_rejections']
                })
            
            print(f"\nRisultato:")
            print(f"  Stato: {result['status']} (gap: {result['gap']:.2%})")
            print(f"  Obiettivo: ${result['objective']:,.0f}")
            print(f"  Tempo totale: {result['total_time']:.1f}s")
            
            if solution.get('summary'):
                print(f"  SP aperti: {result['n_sps_opened']}")
                print(f"  Utilizzo medio: {result['avg_utilization']:.1%}")
                print(f"  Rejections/domanda: {result['total_rejections']/result['total_demand']:.1%}")
            
        except Exception as e:
            print(f"\nERRORE: {e}")
            result = {
                'name': instance['name'],
                'n': instance['n_demand_points'],
                'm': instance['n_sp_locations'],
                'error': str(e),
                'status': 'error'
            }
        
        self.results.append(result)
        self.save_results()  # Salva incrementalmente
        return result
    
    def test_real_cities(self):
        """
        Test con dati reali delle città austriache
        Sezione 5.3 del paper - Real instances
        """
        
        # Controlla se il file OSM esiste
        osm_file = "data/output/osm/austria-latest.osm.pbf"
        if not os.path.exists(osm_file):
            osm_file = "data/osm/austria-latest.osm.pbf"
            
        if not os.path.exists(osm_file):
            print("\n" + "="*60)
            print("DATI REALI NON DISPONIBILI")
            print("="*60)
            print(f"File OSM non trovato in:")
            print(f"  - data/output/osm/austria-latest.osm.pbf")
            print(f"  - data/osm/austria-latest.osm.pbf")
            print("Skippo test città reali")
            return
        
        print("\n" + "="*60)
        print("TEST CON DATI REALI - CITTÀ AUSTRIACHE")
        print("="*60)
        
        try:
            from utils.osm_austria import load_austria_instance
        except ImportError:
            print("Modulo osm_austria non trovato - skippo test città reali")
            return
        
        # Test configurazioni dal paper (Tabella 1)
        # Per velocità, usa subset più piccoli
        city_configs = [
            # (city, max_demand_points, max_sp_candidates, time_limit)
            ("Vienna", 100, 25, 300),    # Subset di Vienna
            ("Graz", 50, 15, 180),       # Subset di Graz
            ("Linz", 30, 10, 120),       # Subset di Linz
        ]
        
        for city, max_n, max_m, timeout in city_configs:
            print(f"\n{'='*40}")
            print(f"Caricamento {city}...")
            print('='*40)
            
            try:
                # Carica dati completi
                start_load = time.time()
                instance_full = load_austria_instance(city, osm_file)
                load_time = time.time() - start_load
                
                print(f"Dati completi caricati in {load_time:.1f}s:")
                print(f"  - {instance_full['n_demand_points']} punti domanda")
                print(f"  - {instance_full['n_sp_locations']} candidati SP")
                
                # Crea subset per test veloce
                instance = {
                    'name': f"{city}_subset",
                    'type': 'real',
                    'city': city,
                    'demand_points': instance_full['demand_points'][:max_n],
                    'sp_locations': instance_full['sp_locations'][:max_m],
                    'demand_rates': instance_full['demand_rates'][:max_n],
                    'service_radius': 300,  # r=300m per città reali
                    'n_demand_points': min(max_n, instance_full['n_demand_points']),
                    'n_sp_locations': min(max_m, instance_full['n_sp_locations']),
                    'total_demand': sum(instance_full['demand_rates'][:max_n]),
                    'time_limit': timeout
                }
                
                print(f"\nTestando subset:")
                print(f"  - {instance['n_demand_points']} punti domanda")
                print(f"  - {instance['n_sp_locations']} candidati SP")
                
                # Testa con capacità del paper per città reali
                self.test_instance(instance, capacities=[30, 60, 90, 120, 150])
                
            except Exception as e:
                print(f"Errore con {city}: {e}")
                import traceback
                traceback.print_exc()
    
    def run_scalability_test(self):
        """Esegue il test completo di scalabilità"""
        
        print("\n" + "="*70)
        print("TEST DI SCALABILITÀ - SP Location and Capacity Problem")
        print("Basato su: Tal Raviv (2023)")
        print("="*70)
        
        # 1. Test con istanze sintetiche
        print("\n\nPARTE 1: ISTANZE SINTETICHE")
        print("="*70)
        instances = self.create_test_instances()
        
        for instance in instances:
            self.test_instance(instance)
        
        # 2. Test con dati reali
        print("\n\nPARTE 2: DATI REALI")
        print("="*70)
        self.test_real_cities()
        
        # 3. Analisi finale
        self.analyze_results()
    
    def save_results(self):
        """Salva i risultati in formato JSON e CSV"""
        
        # JSON
        with open(os.path.join(self.run_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(os.path.join(self.run_dir, 'results.csv'), index=False)
    
    def analyze_results(self):
        """Analizza e visualizza i risultati di scalabilità"""
        
        print("\n\n" + "="*70)
        print("ANALISI RISULTATI")
        print("="*70)
        
        if not self.results:
            print("Nessun risultato da analizzare!")
            return
        
        # Separa risultati validi ed errori
        valid_results = [r for r in self.results if r.get('status') != 'error']
        error_results = [r for r in self.results if r.get('status') == 'error']
        
        print(f"\nRisultati totali: {len(self.results)}")
        print(f"  - Validi: {len(valid_results)}")
        print(f"  - Errori: {len(error_results)}")
        
        if error_results:
            print("\nIstanze con errori:")
            for r in error_results:
                print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
        
        if not valid_results:
            print("\nNessun risultato valido da analizzare!")
            return
        
        # Converti a DataFrame
        df = pd.DataFrame(valid_results)
        
        # Separa risultati sintetici e reali
        df_synth = df[df['type'] == 'synthetic'].copy()
        df_real = df[df['type'] == 'real'].copy()
        
        # Stampa tabella riassuntiva
        print("\n" + "-"*70)
        print("RIEPILOGO RISULTATI")
        print("-"*70)
        
        # Risultati sintetici
        if not df_synth.empty:
            print("\nIstanze sintetiche:")
            print("-"*50)
            cols = ['name', 'n', 'm', 'n_variables', 'total_time', 'gap', 'objective']
            if 'n_sps_opened' in df_synth.columns:
                cols.extend(['n_sps_opened', 'avg_utilization'])
            print(df_synth[cols].to_string(index=False))
        
        # Risultati reali
        if not df_real.empty:
            print("\n\nCittà reali:")
            print("-"*50)
            cols = ['name', 'n', 'm', 'total_time', 'gap', 'objective']
            if 'n_sps_opened' in df_real.columns:
                cols.extend(['n_sps_opened', 'avg_utilization'])
            print(df_real[cols].to_string(index=False))
        
        # Crea grafici
        if not df_synth.empty:
            self.create_plots(df_synth, 'synthetic')
        if not df_real.empty:
            self.create_plots(df_real, 'real')
        
        # Analisi complessità computazionale
        if len(df_synth) >= 3:
            print("\n\nANALISI COMPLESSITÀ (istanze sintetiche):")
            print("-"*50)
            
            # Ordina per dimensione
            df_synth = df_synth.sort_values('n')
            
            # Calcola trend temporale
            x = df_synth['n'].values
            y_time = df_synth['total_time'].values
            
            if len(x) > 2:
                # Fit log-log per stimare complessità
                coeffs = np.polyfit(np.log(x), np.log(y_time), 1)
                print(f"Complessità temporale stimata: O(n^{coeffs[0]:.2f})")
                
                # Crescita variabili
                y_vars = df_synth['n_variables'].values
                growth_rate = (y_vars[-1] - y_vars[0]) / (x[-1] - x[0])
                print(f"Crescita variabili: ~{growth_rate:.1f} per punto domanda")
    
    def create_plots(self, df, plot_type='synthetic'):
        """Crea grafici di analisi scalabilità"""
        
        # Setup
        plt.style.use('seaborn-v0_8-darkgrid')
        suffix = '_synthetic' if plot_type == 'synthetic' else '_real'
        
        # Ordina per dimensione
        df = df.sort_values('n')
        
        # Figura con 4 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Analisi Scalabilità - {"Istanze Sintetiche" if plot_type == "synthetic" else "Città Reali"}', 
                     fontsize=16)
        
        # 1. Tempo vs dimensione
        ax1 = axes[0, 0]
        if 'build_time' in df.columns:
            ax1.plot(df['n'], df['build_time'], 'o-', label='Build', markersize=8, linewidth=2)
        ax1.plot(df['n'], df['solve_time'], 's-', label='Solve', markersize=8, linewidth=2)
        ax1.plot(df['n'], df['total_time'], '^-', label='Total', markersize=8, linewidth=2)
        ax1.set_xlabel('Numero punti domanda (n)')
        ax1.set_ylabel('Tempo (secondi)')
        ax1.set_title('Tempi di Esecuzione')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Dimensione modello
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(df['n'], df['n_variables']/1000, 'o-', color='blue', 
                         label='Variabili (×1000)', markersize=8, linewidth=2)
        line2 = ax2_twin.plot(df['n'], df['n_constraints'], 's-', color='red', 
                             label='Vincoli', markersize=8, linewidth=2)
        
        ax2.set_xlabel('Numero punti domanda (n)')
        ax2.set_ylabel('Variabili (×1000)', color='blue')
        ax2_twin.set_ylabel('Vincoli', color='red')
        ax2.set_title('Dimensione del Modello MILP')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Combina leggende
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Qualità soluzione
        ax3 = axes[1, 0]
        if 'gap' in df.columns and df['gap'].notna().any():
            bars = ax3.bar(df['name'], df['gap'] * 100, color='skyblue', edgecolor='black')
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Target 1%')
            ax3.set_ylabel('Gap di ottimalità (%)')
            ax3.set_title('Qualità della Soluzione')
            ax3.set_xticklabels(df['name'], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Aggiungi valori sopra le barre
            for bar, gap in zip(bars, df['gap']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{gap*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Efficienza rete
        ax4 = axes[1, 1]
        if 'avg_utilization' in df.columns:
            ax4.plot(df['n'], df['avg_utilization'], 'o-', color='green', 
                    markersize=8, linewidth=2, label='Utilizzo medio')
            
            if 'total_rejections' in df.columns and 'total_demand' in df.columns:
                rejection_rate = df['total_rejections'] / df['total_demand'] * 100
                ax4_twin = ax4.twinx()
                ax4_twin.plot(df['n'], rejection_rate, 's-', color='red', 
                             markersize=8, linewidth=2, label='Rejection rate (%)')
                ax4_twin.set_ylabel('Rejection rate (%)', color='red')
                ax4_twin.tick_params(axis='y', labelcolor='red')
            
            ax4.set_xlabel('Numero punti domanda (n)')
            ax4.set_ylabel('Utilizzo medio', color='green')
            ax4.set_title('Efficienza della Rete')
            ax4.set_ylim(0, 1.2)
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='y', labelcolor='green')
        
        plt.tight_layout()
        
        # Salva
        plot_path = os.path.join(self.run_dir, f'scalability_analysis{suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafici salvati in: {plot_path}")
        plt.close()
        
        # Grafico aggiuntivo: trend complessità
        if len(df) >= 3 and plot_type == 'synthetic':
            plt.figure(figsize=(10, 6))
            
            # Plot log-log
            plt.loglog(df['n'], df['total_time'], 'o-', markersize=10, 
                      linewidth=2, label='Tempo effettivo')
            
            # Linee di riferimento
            n_values = df['n'].values
            n_min, n_max = n_values.min(), n_values.max()
            n_ref = np.logspace(np.log10(n_min), np.log10(n_max), 100)
            
            # Normalizza rispetto al primo punto
            t0 = df['total_time'].values[0]
            n0 = n_values[0]
            
            plt.loglog(n_ref, t0 * (n_ref/n0), '--', alpha=0.5, label='O(n)')
            plt.loglog(n_ref, t0 * (n_ref/n0)**2, '--', alpha=0.5, label='O(n²)')
            plt.loglog(n_ref, t0 * (n_ref/n0)**3, '--', alpha=0.5, label='O(n³)')
            
            plt.xlabel('Numero punti domanda (n)')
            plt.ylabel('Tempo totale (secondi)')
            plt.title('Analisi Complessità Computazionale')
            plt.legend()
            plt.grid(True, alpha=0.3, which="both")
            
            trend_path = os.path.join(self.run_dir, 'complexity_trend.png')
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self):
        """Genera report dettagliato in formato testo"""
        
        report_path = os.path.join(self.run_dir, 'report.txt')
        
        with open(report_path, 'w') as f:
            f.write("REPORT TEST DI SCALABILITÀ\n")
            f.write("="*70 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Modello: SP Location and Capacity (Raviv, 2023)\n\n")
            
            f.write("CONFIGURAZIONE TEST:\n")
            f.write("-"*50 + "\n")
            f.write("Parametri modello:\n")
            f.write("  - Probabilità pickup (p): 0.5\n")
            f.write("  - Costo rejection (α): 10\n")
            f.write("  - Costo setup base: 10\n")
            f.write("  - Costo setup variabile: 1/6 per unità capacità\n")
            f.write("  - Breakpoints PWL (K): 12 (piccole), 8 (grandi)\n\n")
            
            f.write("RISULTATI:\n")
            f.write("-"*50 + "\n\n")
            
            # Risultati per istanza
            for r in self.results:
                f.write(f"{r['name']}:\n")
                f.write(f"  Tipo: {r.get('type', 'synthetic')}\n")
                f.write(f"  Dimensione: n={r.get('n', 'N/A')}, m={r.get('m', 'N/A')}\n")
                
                if r.get('status') != 'error':
                    f.write(f"  Modello: {r.get('n_variables', 'N/A')} variabili, ")
                    f.write(f"{r.get('n_constraints', 'N/A')} vincoli\n")
                    f.write(f"  Tempo: {r.get('total_time', 'N/A'):.1f}s ")
                    f.write(f"(build: {r.get('build_time', 'N/A'):.1f}s, ")
                    f.write(f"solve: {r.get('solve_time', 'N/A'):.1f}s)\n")
                    f.write(f"  Stato: {r.get('status', 'N/A')}")
                    if r.get('gap') is not None:
                        f.write(f" (gap: {r['gap']:.2%})")
                    f.write("\n")
                    if r.get('objective'):
                        f.write(f"  Obiettivo: ${r['objective']:,.0f}\n")
                    if r.get('n_sps_opened'):
                        f.write(f"  SP aperti: {r['n_sps_opened']}")
                        f.write(f" (utilizzo medio: {r.get('avg_utilization', 0):.1%})\n")
                else:
                    f.write(f"  ERRORE: {r.get('error', 'Unknown')}\n")
                f.write("\n")
            
            # Analisi complessità
            valid_synth = [r for r in self.results 
                          if r.get('status') != 'error' and r.get('type') == 'synthetic']
            
            if len(valid_synth) >= 3:
                f.write("\nANALISI SCALABILITÀ:\n")
                f.write("-"*50 + "\n")
                
                # Ordina per dimensione
                valid_synth.sort(key=lambda x: x['n'])
                
                # Calcola crescita
                n_values = [r['n'] for r in valid_synth]
                time_values = [r['total_time'] for r in valid_synth]
                
                if len(n_values) > 2:
                    # Stima complessità
                    coeffs = np.polyfit(np.log(n_values), np.log(time_values), 1)
                    f.write(f"Complessità temporale stimata: O(n^{coeffs[0]:.2f})\n")
                    
                    # Tempo per raddoppio dimensione
                    if len(time_values) > 1:
                        time_ratio = time_values[-1] / time_values[0]
                        size_ratio = n_values[-1] / n_values[0]
                        f.write(f"Aumento tempo: {time_ratio:.1f}x per {size_ratio:.1f}x dimensione\n")
        
        print(f"\nReport salvato in: {report_path}")


def main():
    """Funzione principale"""
    
    print("SCALABILITY TEST - Service Points Location and Capacity Problem")
    print("Basato su: Tal Raviv (2023) - Transportation Research Part E")
    
    # Crea tester
    tester = ScalabilityTester()
    
    try:
        # Esegui test
        tester.run_scalability_test()
        
        # Genera report
        tester.generate_report()
        
        print("\n" + "="*70)
        print("TEST DI SCALABILITÀ COMPLETATO!")
        print(f"Risultati salvati in: {tester.run_dir}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrotto dall'utente!")
        tester.save_results()
        tester.generate_report()
    except Exception as e:
        print(f"\n\nERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        tester.save_results()


if __name__ == "__main__":
    main()