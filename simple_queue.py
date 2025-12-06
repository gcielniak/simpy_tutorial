import simpy
import random


def customer(env, name, server, service_time):
    """
    Represents a customer arriving at the queue.
    
    Args:
        env: SimPy environment
        name: Customer identifier
        server: Server resource
        service_time: Time required for service
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

def customer_generator(env, server, arrival_rate, service_rate):
    """
    Generates customers arriving at the queue with specific arrival and service rates."""

    customer_count = 0
    while True:
        # Wait for next arrival (exponentially distributed inter-arrival time)
        yield env.timeout(random.expovariate(arrival_rate))
     
        # Create and start customer process
        customer_count += 1
        env.process(customer(env, f'Customer {customer_count}', server, random.expovariate(service_rate))) # Pass service_time here


ARRIVAL_RATE = 1.0  # Average number of arrivals per time unit (lambda)
SERVICE_RATE = 1.2  # Average number served per time unit (mu)
SIMULATION_TIME = 10.0  # Total simulation time

# Create environment and server resource
env = simpy.Environment()
server = simpy.Resource(env, capacity=1)

# Start monitoring and customer generation processes
env.process(customer_generator(env, server, ARRIVAL_RATE, SERVICE_RATE))

# Run simulation
env.run(until=SIMULATION_TIME)

