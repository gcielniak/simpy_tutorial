import simpy
import random


class QueueSimulation(simpy.Environment):
    """
    A queue simulation class derived from simpy.Environment.
    
    This class simulates a single-server queue with exponentially distributed
    arrival and service times.
    """
    
    def __init__(self, arrival_rate, service_rate, server_capacity=1):
        """
        Initialize the queue simulation.
        
        Args:
            arrival_rate: Average number of arrivals per time unit (lambda)
            service_rate: Average number served per time unit (mu)
            server_capacity: Number of servers (default=1)
        """
        super().__init__()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server = simpy.Resource(self, capacity=server_capacity)
        self.customer_count = 0
    
    def customer(self, name, service_time):
        """
        Represents a customer arriving at the queue.
        
        Args:
            name: Customer identifier
            service_time: Time required for service
        """
        arrival_time = self.now
        print(f'{arrival_time:.2f}: {name} arrives')
        
        with self.server.request() as request:
            # Wait in queue
            yield request
            wait_time = self.now - arrival_time
            print(f'{self.now:.2f}: {name} enters service (waited {wait_time:.2f})')
            
            # Being served
            yield self.timeout(service_time)
            print(f'{self.now:.2f}: {name} departs')
    
    def customer_generator(self):
        """
        Generates customers arriving at the queue with exponentially 
        distributed inter-arrival and service times.
        """
        while True:
            # Wait for next arrival (exponentially distributed inter-arrival time)
            yield self.timeout(random.expovariate(self.arrival_rate))
         
            # Create and start customer process
            self.customer_count += 1
            service_time = random.expovariate(self.service_rate)
            self.process(self.customer(f'Customer {self.customer_count}', service_time))
    
    def run_simulation(self, simulation_time):
        """
        Start and run the simulation.
        
        Args:
            simulation_time: Total simulation time
        """
        # Start customer generation process
        self.process(self.customer_generator())
        
        # Run simulation
        self.run(until=simulation_time)


if __name__ == '__main__':
    # Simulation parameters
    ARRIVAL_RATE = 1.0  # Average number of arrivals per time unit (lambda)
    SERVICE_RATE = 1.2  # Average number served per time unit (mu)
    SIMULATION_TIME = 10.0  # Total simulation time
    
    # Create and run simulation
    sim = QueueSimulation(ARRIVAL_RATE, SERVICE_RATE)
    sim.run_simulation(SIMULATION_TIME)
