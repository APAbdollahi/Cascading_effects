import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import random
from datetime import datetime, timedelta

class CascadingExclusionSimulation:
    def __init__(self, population_size=10000, simulation_months=60):
        self.population_size = population_size
        self.simulation_months = simulation_months
        self.time_step = 0.1  # 0.1 month increments
        
        # System definitions
        self.systems = ['credit', 'employment', 'housing', 'education', 'services']
        
        # Initialize population and tracking
        self.population = self._generate_population()
        self.events = []
        self.results = defaultdict(list)
        
        # Core parameters
        self._set_parameters()
        
    def _generate_population(self):
        """Generate synthetic population with demographics"""
        pop = []
        
        # Demographic distributions
        demographics = {
            'white': 0.60, 'black': 0.13, 'hispanic': 0.18, 
            'asian': 0.06, 'other': 0.03
        }
        
        immigration_status = {
            'native_born': 0.87, 'naturalized': 0.06, 
            'permanent_resident': 0.04, 'recent_immigrant': 0.03
        }
        
        income_levels = {
            'low': 0.25, 'lower_mid': 0.25, 'mid': 0.30, 
            'upper_mid': 0.15, 'high': 0.05
        }
        
        for i in range(self.population_size):
            person = {
                'id': i,
                'race': np.random.choice(list(demographics.keys()), 
                                       p=list(demographics.values())),
                'immigration': np.random.choice(list(immigration_status.keys()), 
                                              p=list(immigration_status.values())),
                'income_level': np.random.choice(list(income_levels.keys()), 
                                               p=list(income_levels.values())),
                'system_history': {system: [] for system in self.systems},
                'cascade_count': 0,
                'total_rejections': 0,
                'last_rejection_time': {system: -np.inf for system in self.systems}
            }
            
            # Add risk factors based on demographics
            person['risk_factors'] = self._assign_risk_factors(person)
            pop.append(person)
            
        return pop
    
    def _assign_risk_factors(self, person):
        """Assign additional risk factors based on demographics"""
        factors = {}
        
        # Higher risk factors for marginalized groups
        if person['race'] in ['black', 'hispanic']:
            factors['high_risk_zip'] = np.random.random() < 0.4
            factors['address_instability'] = np.random.random() < 0.3
        else:
            factors['high_risk_zip'] = np.random.random() < 0.15
            factors['address_instability'] = np.random.random() < 0.1
            
        if person['immigration'] == 'recent_immigrant':
            factors['thin_file'] = True
            factors['foreign_address_history'] = True
            factors['documentation_complex'] = np.random.random() < 0.7
        else:
            factors['thin_file'] = np.random.random() < 0.1
            factors['foreign_address_history'] = False
            factors['documentation_complex'] = np.random.random() < 0.05
            
        return factors
    
    def _set_parameters(self):
        """Set all simulation parameters"""
        
        # Baseline exclusion rates by system and demographics
        self.baseline_rates = {
            'credit': {
                'white': 0.08, 'black': 0.22, 'hispanic': 0.18, 
                'asian': 0.09, 'other': 0.15
            },
            'employment': {
                'white': 0.05, 'black': 0.15, 'hispanic': 0.12, 
                'asian': 0.06, 'other': 0.10
            },
            'housing': {
                'white': 0.06, 'black': 0.19, 'hispanic': 0.15, 
                'asian': 0.07, 'other': 0.12
            },
            'education': {
                'white': 0.04, 'black': 0.13, 'hispanic': 0.11, 
                'asian': 0.05, 'other': 0.08
            },
            'services': {
                'white': 0.07, 'black': 0.16, 'hispanic': 0.14, 
                'asian': 0.08, 'other': 0.11
            }
        }
        
        # Immigration status multipliers
        self.immigration_multipliers = {
            'native_born': 1.0,
            'naturalized': 1.2,
            'permanent_resident': 1.5,
            'recent_immigrant': 2.8
        }
        
        # Cross-system excitation matrix (how rejection in j affects i)
        self.excitation_matrix = {
            'credit':     {'credit': 0.0, 'employment': 0.65, 'housing': 0.75, 'education': 0.45, 'services': 0.55},
            'employment': {'credit': 0.70, 'employment': 0.0, 'housing': 0.80, 'education': 0.40, 'services': 0.60},
            'housing':    {'credit': 0.85, 'employment': 0.72, 'housing': 0.0, 'education': 0.68, 'services': 0.73},
            'education':  {'credit': 0.35, 'employment': 0.58, 'housing': 0.48, 'education': 0.0, 'services': 0.52},
            'services':   {'credit': 0.58, 'employment': 0.63, 'housing': 0.71, 'education': 0.46, 'services': 0.0}
        }
        
        # Decay rates (memory length) for each system
        self.decay_rates = {
            'credit': 0.08,      # ~8.7 month half-life
            'employment': 0.12,   # ~5.8 month half-life
            'housing': 0.15,     # ~4.6 month half-life
            'education': 0.05,   # ~13.9 month half-life
            'services': 0.10     # ~6.9 month half-life
        }
        
        # Application rates (how often people apply to each system)
        self.application_rates = {
            'credit': 1.2/12,        # Per month
            'employment': 0.8/12,
            'housing': 0.6/12,
            'education': 0.3/12,
            'services': 2.1/12
        }
        
        # Risk factor multipliers
        self.risk_multipliers = {
            'thin_file': 2.5,
            'foreign_address_history': 1.8,
            'high_risk_zip': 1.4,
            'address_instability': 1.5,
            'documentation_complex': 1.9
        }
    
    def _calculate_hawkes_intensity(self, person, system, current_time):
        """Calculate Hawkes process intensity for person/system at current time"""
        
        # Base intensity
        base_rate = self.baseline_rates[system][person['race']]
        base_rate *= self.immigration_multipliers[person['immigration']]
        
        # Apply risk factor multipliers
        for factor, multiplier in self.risk_multipliers.items():
            if person['risk_factors'].get(factor, False):
                base_rate *= multiplier
        
        # Add excitation from past rejections
        excitation = 0
        for other_system in self.systems:
            if other_system != system:
                # Get recent rejections from other systems
                rejections = person['system_history'][other_system]
                for rejection_time in rejections:
                    time_diff = current_time - rejection_time
                    if time_diff > 0:  # Only past events
                        # Exponential decay kernel
                        decay = np.exp(-self.decay_rates[other_system] * time_diff)
                        excitation += self.excitation_matrix[system][other_system] * decay
        
        return base_rate + excitation
    
    def _process_application(self, person, system, current_time):
        """Process a single application and determine outcome"""
        
        # Calculate rejection probability using Hawkes intensity
        intensity = self._calculate_hawkes_intensity(person, system, current_time)
        
        # Convert intensity to probability (assuming Poisson process approximation)
        rejection_prob = 1 - np.exp(-intensity * self.time_step)
        
        # Make decision
        rejected = np.random.random() < rejection_prob
        
        # Record outcome
        event = {
            'person_id': person['id'],
            'system': system,
            'time': current_time,
            'rejected': rejected,
            'intensity': intensity,
            'race': person['race'],
            'immigration': person['immigration']
        }
        
        self.events.append(event)
        
        # Update person's history
        if rejected:
            person['system_history'][system].append(current_time)
            person['total_rejections'] += 1
            person['last_rejection_time'][system] = current_time
            
            # Check for cascade (rejection within 6 months of another rejection)
            recent_rejections = 0
            for sys_history in person['system_history'].values():
                recent_rejections += sum(1 for t in sys_history if current_time - t <= 6)
            
            if recent_rejections >= 2:
                person['cascade_count'] += 1
        
        return rejected
    
    def run_simulation(self):
        """Run the full simulation"""
        print("Starting Cascading Exclusion Simulation...")
        print(f"Population: {self.population_size}")
        print(f"Duration: {self.simulation_months} months")
        
        # Time loop
        for month in np.arange(0, self.simulation_months, self.time_step):
            
            # Progress indicator
            if int(month) % 12 == 0 and month > 0:
                print(f"Year {int(month/12)} completed...")
            
            # For each person, determine if they apply to any systems
            for person in self.population:
                for system in self.systems:
                    # Poisson process for applications
                    apply_prob = self.application_rates[system] * self.time_step
                    
                    if np.random.random() < apply_prob:
                        self._process_application(person, system, month)
        
        print("Simulation completed!")
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze simulation results and generate metrics"""
        
        # Convert events to DataFrame for easier analysis
        df = pd.DataFrame(self.events)
        
        if len(df) == 0:
            print("No events generated!")
            return
        
        # Basic statistics
        total_applications = len(df)
        total_rejections = df['rejected'].sum()
        overall_rejection_rate = total_rejections / total_applications
        
        print(f"\n=== SIMULATION RESULTS ===")
        print(f"Total applications: {total_applications:,}")
        print(f"Total rejections: {total_rejections:,}")
        print(f"Overall rejection rate: {overall_rejection_rate:.3f}")
        
        # Demographic breakdown
        print(f"\n=== REJECTION RATES BY DEMOGRAPHICS ===")
        demo_analysis = df.groupby('race').agg({
            'rejected': ['count', 'sum', 'mean']
        }).round(3)
        demo_analysis.columns = ['Applications', 'Rejections', 'Rejection_Rate']
        print(demo_analysis)
        
        # Immigration status breakdown
        print(f"\n=== REJECTION RATES BY IMMIGRATION STATUS ===")
        immig_analysis = df.groupby('immigration').agg({
            'rejected': ['count', 'sum', 'mean']
        }).round(3)
        immig_analysis.columns = ['Applications', 'Rejections', 'Rejection_Rate']
        print(immig_analysis)
        
        # System-specific analysis
        print(f"\n=== REJECTION RATES BY SYSTEM ===")
        system_analysis = df.groupby('system').agg({
            'rejected': ['count', 'sum', 'mean']
        }).round(3)
        system_analysis.columns = ['Applications', 'Rejections', 'Rejection_Rate']
        print(system_analysis)
        
        # Cascade analysis
        cascade_people = [p for p in self.population if p['cascade_count'] > 0]
        cascade_rate = len(cascade_people) / self.population_size
        
        print(f"\n=== CASCADE ANALYSIS ===")
        print(f"People experiencing cascades: {len(cascade_people):,}")
        print(f"Overall cascade rate: {cascade_rate:.3f}")
        
        # Cascade rates by demographics
        cascade_by_race = {}
        for race in ['white', 'black', 'hispanic', 'asian', 'other']:
            race_pop = [p for p in self.population if p['race'] == race]
            race_cascades = [p for p in race_pop if p['cascade_count'] > 0]
            if len(race_pop) > 0:
                cascade_by_race[race] = len(race_cascades) / len(race_pop)
        
        print(f"\n=== CASCADE RATES BY RACE ===")
        for race, rate in cascade_by_race.items():
            print(f"{race}: {rate:.3f}")
        
        # Store results for plotting
        self.results = {
            'events_df': df,
            'cascade_rate': cascade_rate,
            'cascade_by_race': cascade_by_race,
            'overall_rejection_rate': overall_rejection_rate
        }
    
    def plot_results(self):
        """Generate visualization of results"""
        
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        df = self.results['events_df']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Rejection rates by race and system
        rejection_by_race_system = df.groupby(['race', 'system'])['rejected'].mean().unstack()
        rejection_by_race_system.plot(kind='bar', ax=axes[0,0], rot=45)
        axes[0,0].set_title('Rejection Rates by Race and System')
        axes[0,0].set_ylabel('Rejection Rate')
        axes[0,0].legend(title='System', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Cascade rates by race
        cascade_data = list(self.results['cascade_by_race'].values())
        cascade_labels = list(self.results['cascade_by_race'].keys())
        axes[0,1].bar(cascade_labels, cascade_data, color='red', alpha=0.7)
        axes[0,1].set_title('Cascade Rates by Race')
        axes[0,1].set_ylabel('Cascade Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Time series of rejections
        df['month'] = (df['time']).astype(int)
        monthly_rejections = df[df['rejected']].groupby('month').size()
        axes[1,0].plot(monthly_rejections.index, monthly_rejections.values)
        axes[1,0].set_title('Monthly Rejection Counts')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Number of Rejections')
        
        # Plot 4: Intensity distribution
        axes[1,1].hist(df['intensity'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Distribution of Hawkes Intensities')
        axes[1,1].set_xlabel('Intensity')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis plot
        self._plot_cascade_demonstration()
    
    def _plot_cascade_demonstration(self):
        """Create specific plot showing cascade effect"""
        
        df = self.results['events_df']
        
        # Find examples of people who experienced cascades
        cascade_people = [p for p in self.population if p['cascade_count'] > 0]
        
        if len(cascade_people) == 0:
            print("No cascades found to demonstrate")
            return
        
        # Take first few cascade examples
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, person in enumerate(cascade_people[:4]):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            
            # Get this person's rejection timeline
            person_events = df[df['person_id'] == person['id']]
            rejected_events = person_events[person_events['rejected']]
            
            if len(rejected_events) > 0:
                # Plot rejection timeline
                systems_numeric = {sys: j for j, sys in enumerate(self.systems)}
                y_values = [systems_numeric[sys] for sys in rejected_events['system']]
                
                axes[row, col].scatter(rejected_events['time'], y_values, 
                                     c='red', s=100, alpha=0.7)
                axes[row, col].set_yticks(list(systems_numeric.values()))
                axes[row, col].set_yticklabels(list(systems_numeric.keys()))
                axes[row, col].set_xlabel('Time (months)')
                axes[row, col].set_title(f'Person {person["id"]} ({person["race"]}) - {len(rejected_events)} rejections')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.suptitle('Examples of Cascading Exclusion Patterns', fontsize=16)
        plt.tight_layout()
        plt.show()

# Run the simulation
if __name__ == "__main__":
    # Create and run simulation
    sim = CascadingExclusionSimulation(population_size=5000, simulation_months=36)
    sim.run_simulation()
    sim.plot_results()
    
    # Print key findings
    print(f"\n=== KEY FINDINGS ===")
    
    # Calculate disparate impact ratios
    cascade_by_race = sim.results['cascade_by_race']
    if 'white' in cascade_by_race and 'black' in cascade_by_race:
        disparity_ratio = cascade_by_race['black'] / cascade_by_race['white'] if cascade_by_race['white'] > 0 else float('inf')
        print(f"Black-to-White cascade ratio: {disparity_ratio:.2f}")
    
    # Show most vulnerable populations
    sorted_cascades = sorted(cascade_by_race.items(), key=lambda x: x[1], reverse=True)
    print(f"Most vulnerable population: {sorted_cascades[0][0]} ({sorted_cascades[0][1]:.3f} cascade rate)")
    
    print(f"\nThis demonstrates how small initial biases create large systemic disparities through cascading effects.")