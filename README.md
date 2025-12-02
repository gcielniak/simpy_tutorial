# SimPy Tutorial

Simulation tutorials using the SimPy discrete-event simulation library.

## Single Queue, Multiple Servers Simulation

This example demonstrates an M/M/c queueing system simulation where:
- Customers arrive according to a Poisson process (exponential inter-arrival times)
- There is a single queue for all customers
- Multiple servers process customers from the queue in FIFO order
- Service times follow an exponential distribution

### Features

- Configurable number of servers, arrival rate, and service rate
- Tracks key performance metrics:
  - Customer waiting times
  - Queue length over time
  - Server utilization
- Generates visualization of simulation results

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the simulation:

```bash
python single_queue_multi_server.py
```

### Configuration

You can modify the simulation parameters in the `main()` function:

```python
config = SimulationConfig(
    num_servers=3,           # Number of servers
    arrival_rate=5.0,        # Average arrivals per time unit (λ)
    service_rate=2.0,        # Average service rate per server (μ)
    simulation_time=100.0,   # Total simulation time
    random_seed=42           # Random seed for reproducibility
)
```

### Output

The simulation produces:
1. Console output with summary statistics
2. A visualization saved as `simulation_results.png` showing:
   - Queue length over time
   - Wait time distribution
   - Wait time vs arrival time
   - Summary statistics panel

### Example Output

![Simulation Results](simulation_results.png)

### Queueing Theory Background

The M/M/c queue is characterized by:
- **M**: Markovian (exponential) inter-arrival times
- **M**: Markovian (exponential) service times  
- **c**: Number of servers

Key metrics:
- **Traffic intensity (ρ)**: λ / (c × μ) - should be < 1 for stable system
- **Utilization**: Fraction of time servers are busy
