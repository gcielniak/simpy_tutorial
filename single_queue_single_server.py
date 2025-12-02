"""
Single Queue Single Server Simulation using SimPy

A minimal example demonstrating basic SimPy queue functionality with:
- Customers arriving according to a Poisson process
- A single queue
- A single server
- Exponential service times
"""

import simpy
import random
import matplotlib.pyplot as plt


def customer(env, name, server, service_time, stats):
    """
    Represents a customer arriving at the queue.
    
    Args:
        env: SimPy environment
        name: Customer identifier
        server: Server resource
        service_time: Time required for service
        stats: Dictionary to store statistics
    """
    arrival_time = env.now
    print(f'{arrival_time:.2f}: {name} arrives')
    
    with server.request() as request:
        # Wait in queue
        yield request
        wait_time = env.now - arrival_time
        print(f'{env.now:.2f}: {name} enters service (waited {wait_time:.2f})')
        
        # Being served
        yield env.timeout(service_time)
        print(f'{env.now:.2f}: {name} departs')


def customer_generator(env, server, arrival_rate, service_rate, stats):
    """
    Generates customers arriving at the queue.
    
    Args:
        env: SimPy environment
        server: Server resource
        arrival_rate: Average number of arrivals per time unit (lambda)
        service_rate: Average number served per time unit (mu)
        stats: Dictionary to store statistics
    """
    customer_count = 0
    while True:
        # Wait for next arrival (exponentially distributed inter-arrival time)
        inter_arrival_time = random.expovariate(arrival_rate)
        yield env.timeout(inter_arrival_time)
        
        # Generate service time (exponentially distributed)
        service_time = random.expovariate(service_rate)
        
        # Create and start customer process
        customer_count += 1
        env.process(customer(env, f'Customer {customer_count}', server, service_time, stats))


def monitor(env, server, stats, interval=0.1):
    """
    Monitor server utilization over time.
    
    Args:
        env: SimPy environment
        server: Server resource to monitor
        stats: Dictionary to store statistics
        interval: Time interval between samples
    """
    while True:
        # Record current time and server utilization
        stats['times'].append(env.now)
        # Server is busy if count > 0 (capacity - count = number busy)
        utilization = 1 if server.count > 0 else 0
        stats['utilization'].append(utilization)
        
        yield env.timeout(interval)


def plot_utilization(stats):
    """
    Plot server utilization over time.
    
    Args:
        stats: Dictionary containing simulation statistics
    """
    plt.figure(figsize=(10, 6))
    plt.step(stats['times'], stats['utilization'], where='post', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Server Utilization')
    plt.title('Server Utilization Over Time')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_simulation():
    """Run the single queue, single server simulation."""
    # Simulation parameters
    ARRIVAL_RATE = 2.0      # customers per time unit (lambda)
    SERVICE_RATE = 3.0      # customers per time unit (mu)
    SIMULATION_TIME = 20.0  # total simulation time
    RANDOM_SEED = 42
    
    print('Single Queue, Single Server Simulation')
    print('=' * 50)
    print(f'Arrival rate (λ): {ARRIVAL_RATE}')
    print(f'Service rate (μ): {SERVICE_RATE}')
    print(f'Utilization (ρ): {ARRIVAL_RATE/SERVICE_RATE:.2f}')
    print('=' * 50)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Create environment and server resource
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    
    # Initialize statistics
    stats = {
        'times': [],
        'utilization': []
    }
    
    # Start monitoring and customer generation processes
    env.process(monitor(env, server, stats))
    env.process(customer_generator(env, server, ARRIVAL_RATE, SERVICE_RATE, stats))
    
    # Run simulation
    env.run(until=SIMULATION_TIME)
    
    print('=' * 50)
    print('Simulation complete')
    
    # Calculate and display average utilization
    avg_utilization = sum(stats['utilization']) / len(stats['utilization'])
    print(f'Average server utilization: {avg_utilization:.3f}')
    print('=' * 50)
    
    # Plot results
    plot_utilization(stats)


if __name__ == '__main__':
    run_simulation()
