"""
Workload Generator
-------------------
Generates synthetic CPU process workloads for the scheduling simulator.
 
Each process has:
- id: unique identifier (P1, P2, ...)
- arrival_time: when the process enters the ready queue (Poisson-sampled)
- burst_time: how much CPU time the process needs (exponential-sampled)
- priority: integer priority level (1 = highest, 10 = lowest)
 
Using a fixed random seed makes workloads reproducible, so FCFS, RR,
Priority, and the RL agent can all be evaluated on identical workloads.
"""
 
from dataclasses import dataclass
import numpy as np
 
 
@dataclass
class Process:
    id: str
    arrival_time: int
    burst_time: int
    priority: int
 
    def __repr__(self):
        return (f"Process({self.id}, arrival={self.arrival_time}, "
                f"burst={self.burst_time}, priority={self.priority})")
 
 
def generate_workload(num_processes: int = 10, seed: int = 42,
                       arrival_rate: float = 2.0,
                       mean_burst_time: float = 5.0,
                       min_burst_time: int = 1,
                       priority_range: tuple = (1, 10)) -> list[Process]:
    """
    Generate a synthetic workload of processes.
 
    Parameters
    ----------
    num_processes : int
        Number of processes to generate.
    seed : int
        Random seed for reproducibility. Same seed -> same workload.
    arrival_rate : float
        Average number of process arrivals per time unit (Poisson lambda).
        Higher = processes arrive closer together.
    mean_burst_time : float
        Mean CPU burst time (used as the scale for the exponential distribution).
    min_burst_time : int
        Minimum burst time, to avoid zero-length or negative bursts after rounding.
    priority_range : tuple
        (min_priority, max_priority) inclusive range for random priority assignment.
 
    Returns
    -------
    list[Process]
        List of Process objects sorted by arrival_time.
    """
    rng = np.random.default_rng(seed)
 
    # Inter-arrival times sampled from a Poisson process.
    # Cumulative sum gives increasing arrival times.
    inter_arrival_times = rng.poisson(lam=arrival_rate, size=num_processes)
    arrival_times = np.cumsum(inter_arrival_times)
 
    # Burst times sampled from an exponential distribution, rounded to
    # whole time units, with a minimum floor so no process has 0 burst time.
    burst_times = rng.exponential(scale=mean_burst_time, size=num_processes)
    burst_times = np.maximum(min_burst_time, np.round(burst_times)).astype(int)
 
    # Priorities sampled uniformly from the given range (inclusive).
    priorities = rng.integers(priority_range[0], priority_range[1] + 1, size=num_processes)
 
    processes = [
        Process(
            id=f"P{i+1}",
            arrival_time=int(arrival_times[i]),
            burst_time=int(burst_times[i]),
            priority=int(priorities[i]),
        )
        for i in range(num_processes)
    ]
 
    # Sort by arrival time so the simulator can process them in order.
    processes.sort(key=lambda p: p.arrival_time)
    return processes
 
 
if __name__ == "__main__":
    # Quick manual test: generate and print a small workload.
    workload = generate_workload(num_processes=500, seed=42)
    print(f"Generated {len(workload)} processes:\n")
    for p in workload:
        print(p)
 