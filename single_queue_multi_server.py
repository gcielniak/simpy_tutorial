"""
Single Queue Multiple Servers Simulation using SimPy

This script simulates an M/M/c queueing system where:
- Customers arrive according to a Poisson process
- There is a single queue for all customers
- Multiple servers process customers from the queue
- Service times follow an exponential distribution

The simulation tracks and visualizes key performance metrics:
- Queue length over time
- Customer waiting times
- Server utilization
"""

import random
import simpy
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    num_servers: int = 3           # Number of servers
    arrival_rate: float = 5.0      # Average arrivals per time unit (lambda)
    service_rate: float = 2.0      # Average service rate per server (mu)
    simulation_time: float = 100.0  # Total simulation time
    random_seed: int = 42          # Random seed for reproducibility


@dataclass
class Statistics:
    """Container for simulation statistics."""
    wait_times: list = field(default_factory=list)
    queue_lengths: list = field(default_factory=list)
    queue_times: list = field(default_factory=list)
    utilization_times: list = field(default_factory=list)
    utilization_values: list = field(default_factory=list)


class QueueSimulation:
    """
    Single queue, multiple server simulation.
    
    This class implements an M/M/c queue where customers arrive,
    wait in a single queue, and are served by one of multiple servers.
    """
    
    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        """
        Initialize the simulation.
        
        Args:
            env: SimPy environment
            config: Simulation configuration parameters
        """
        self.env = env
        self.config = config
        self.servers = simpy.Resource(env, capacity=config.num_servers)
        self.stats = Statistics()
        self.customer_count = 0
    
    def customer(self, customer_id: int):
        """
        Customer process - arrives, waits in queue, gets served, and leaves.
        
        Args:
            customer_id: Unique identifier for the customer
        """
        arrival_time = self.env.now
        
        # Record queue length when customer arrives
        queue_length = len(self.servers.queue)
        self.stats.queue_lengths.append((arrival_time, queue_length))
        
        # Request a server (wait in queue if all busy)
        with self.servers.request() as request:
            yield request
            
            # Customer got a server - record wait time
            wait_time = self.env.now - arrival_time
            self.stats.wait_times.append(wait_time)
            self.stats.queue_times.append((arrival_time, wait_time))
            
            # Service time (exponentially distributed)
            service_time = random.expovariate(self.config.service_rate)
            yield self.env.timeout(service_time)
    
    def customer_arrivals(self):
        """Generate customer arrivals according to a Poisson process."""
        while True:
            # Inter-arrival time (exponentially distributed)
            inter_arrival = random.expovariate(self.config.arrival_rate)
            yield self.env.timeout(inter_arrival)
            
            self.customer_count += 1
            self.env.process(self.customer(self.customer_count))
    
    def monitor_queue(self, interval: float = 0.5):
        """
        Periodically monitor and record queue statistics.
        
        Args:
            interval: Time between monitoring events
        """
        while True:
            yield self.env.timeout(interval)
            queue_length = len(self.servers.queue)
            self.stats.queue_lengths.append((self.env.now, queue_length))
    
    def monitor_utilization(self, interval: float = 0.1):
        """
        Periodically monitor and record server utilization.
        
        Args:
            interval: Time between monitoring events
        """
        while True:
            # Calculate utilization: number of busy servers / total servers
            busy_servers = self.servers.count
            utilization = busy_servers / self.config.num_servers
            self.stats.utilization_times.append(self.env.now)
            self.stats.utilization_values.append(utilization)
            yield self.env.timeout(interval)


def run_simulation(config: SimulationConfig) -> tuple:
    """
    Run the queue simulation with the given configuration.
    
    Args:
        config: Simulation configuration parameters
        
    Returns:
        Tuple of (statistics, customer_count)
    """
    # Set random seed for reproducibility
    random.seed(config.random_seed)
    
    # Create environment and simulation
    env = simpy.Environment()
    simulation = QueueSimulation(env, config)
    
    # Start processes
    env.process(simulation.customer_arrivals())
    env.process(simulation.monitor_queue())
    env.process(simulation.monitor_utilization())
    
    # Run simulation
    env.run(until=config.simulation_time)
    
    return simulation.stats, simulation.customer_count


def calculate_metrics(stats: Statistics, config: SimulationConfig, 
                      customer_count: int) -> dict:
    """
    Calculate summary metrics from the simulation statistics.
    
    Args:
        stats: Collected statistics from simulation
        config: Simulation configuration
        customer_count: Total number of customers processed
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        'total_customers': customer_count,
        'avg_wait_time': sum(stats.wait_times) / len(stats.wait_times) 
                         if stats.wait_times else 0,
        'max_wait_time': max(stats.wait_times) if stats.wait_times else 0,
        'avg_queue_length': sum(q[1] for q in stats.queue_lengths) / 
                           len(stats.queue_lengths) if stats.queue_lengths else 0,
        'max_queue_length': max(q[1] for q in stats.queue_lengths) 
                           if stats.queue_lengths else 0,
        'utilization': config.arrival_rate / (config.num_servers * 
                                              config.service_rate),
    }
    return metrics


def visualize_results(stats: Statistics, config: SimulationConfig, 
                      metrics: dict, output_file: str = 'simulation_results.png'):
    """
    Create visualizations of the simulation results.
    
    Args:
        stats: Collected statistics from simulation
        config: Simulation configuration
        metrics: Calculated metrics dictionary
        output_file: Path to save the visualization image
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f'Single Queue, {config.num_servers} Servers Simulation Results',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Queue Length Over Time
    ax1 = axes[0, 0]
    times = [q[0] for q in stats.queue_lengths]
    lengths = [q[1] for q in stats.queue_lengths]
    ax1.plot(times, lengths, 'b-', alpha=0.7, linewidth=0.8)
    ax1.fill_between(times, lengths, alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Queue Length')
    ax1.set_title('Queue Length Over Time')
    ax1.axhline(y=metrics['avg_queue_length'], color='r', linestyle='--', 
                label=f"Avg: {metrics['avg_queue_length']:.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wait Time Distribution
    ax2 = axes[0, 1]
    ax2.hist(stats.wait_times, bins=30, edgecolor='black', alpha=0.7, 
             color='steelblue')
    ax2.axvline(x=metrics['avg_wait_time'], color='r', linestyle='--', 
                linewidth=2, label=f"Avg: {metrics['avg_wait_time']:.2f}")
    ax2.set_xlabel('Wait Time')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Customer Wait Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wait Time Over Time
    ax3 = axes[1, 0]
    arrival_times = [q[0] for q in stats.queue_times]
    wait_times = [q[1] for q in stats.queue_times]
    ax3.scatter(arrival_times, wait_times, alpha=0.5, s=10, c='steelblue')
    ax3.set_xlabel('Arrival Time')
    ax3.set_ylabel('Wait Time')
    ax3.set_title('Wait Time vs Arrival Time')
    ax3.axhline(y=metrics['avg_wait_time'], color='r', linestyle='--',
                label=f"Avg: {metrics['avg_wait_time']:.2f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Server Utilization Over Time
    ax4 = axes[1, 1]
    if stats.utilization_times:
        ax4.plot(stats.utilization_times, stats.utilization_values, 
                'g-', alpha=0.7, linewidth=1)
        ax4.fill_between(stats.utilization_times, stats.utilization_values, 
                        alpha=0.3, color='green')
        avg_utilization = sum(stats.utilization_values) / len(stats.utilization_values)
        ax4.axhline(y=avg_utilization, color='r', linestyle='--',
                   label=f"Avg: {avg_utilization:.3f}")
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Utilization (Busy Servers / Total Servers)')
        ax4.set_title('Server Utilization Over Time')
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Summary Statistics (Text)
    ax5 = axes[2, 0]
    ax5.axis('off')
    
    # Calculate theoretical values for M/M/c queue
    lambda_rate = config.arrival_rate
    mu_rate = config.service_rate
    c = config.num_servers
    rho = lambda_rate / (c * mu_rate)  # Traffic intensity
    
    # Calculate actual average utilization from monitoring
    actual_utilization = (sum(stats.utilization_values) / len(stats.utilization_values) 
                         if stats.utilization_values else 0)
    
    summary_text = f"""
    SIMULATION PARAMETERS
    ─────────────────────────
    Number of Servers: {config.num_servers}
    Arrival Rate (λ): {config.arrival_rate} customers/time unit
    Service Rate (μ): {config.service_rate} customers/time unit/server
    Simulation Time: {config.simulation_time} time units
    
    RESULTS
    ─────────────────────────
    Total Customers Served: {metrics['total_customers']}
    
    WAIT TIME
    Average: {metrics['avg_wait_time']:.3f} time units
    Maximum: {metrics['max_wait_time']:.3f} time units
    
    QUEUE LENGTH
    Average: {metrics['avg_queue_length']:.2f} customers
    Maximum: {metrics['max_queue_length']} customers
    
    SYSTEM METRICS
    Traffic Intensity (ρ): {rho:.3f}
    Theoretical Utilization: {rho * 100:.1f}%
    Actual Avg Utilization: {actual_utilization * 100:.1f}%
    """
    
    ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Hide the empty subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to '{output_file}'")


def print_summary(metrics: dict, config: SimulationConfig):
    """Print a summary of the simulation results to console."""
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Number of servers: {config.num_servers}")
    print(f"  - Arrival rate (λ): {config.arrival_rate} customers/time unit")
    print(f"  - Service rate (μ): {config.service_rate} per server/time unit")
    print(f"  - Simulation time: {config.simulation_time} time units")
    
    print(f"\nResults:")
    print(f"  - Total customers served: {metrics['total_customers']}")
    print(f"  - Average wait time: {metrics['avg_wait_time']:.3f} time units")
    print(f"  - Maximum wait time: {metrics['max_wait_time']:.3f} time units")
    print(f"  - Average queue length: {metrics['avg_queue_length']:.2f} customers")
    print(f"  - Maximum queue length: {metrics['max_queue_length']} customers")
    print(f"  - System utilization: {metrics['utilization']*100:.1f}%")
    print("="*60 + "\n")


def main():
    """Main function to run the simulation."""
    # Create configuration
    config = SimulationConfig(
        num_servers=3,
        arrival_rate=5.0,
        service_rate=2.0,
        simulation_time=100.0,
        random_seed=42
    )
    
    print("Starting Single Queue, Multiple Servers Simulation...")
    print(f"Configuration: {config.num_servers} servers, "
          f"λ={config.arrival_rate}, μ={config.service_rate}")
    
    # Run simulation
    stats, customer_count = run_simulation(config)
    
    # Calculate metrics
    metrics = calculate_metrics(stats, config, customer_count)
    
    # Print summary
    print_summary(metrics, config)
    
    # Visualize results
    visualize_results(stats, config, metrics)
    
    print("Simulation complete!")


if __name__ == "__main__":
    main()
