
from simulator import CPUSimulator
 
 
def run_round_robin(sim: CPUSimulator, quantum: int = 3) -> dict:
    """
    Run the Round Robin policy on a (freshly reset) simulator until
    all processes finish, then return the performance metrics.
 
    Parameters
    ----------
    quantum : int
        Maximum time units a process may run before being preempted
        and sent to the back of the queue if unfinished.
    """
    sim.reset()
 
    while not sim.is_done():
        if not sim.ready_queue:
            sim.time = sim._pending[0].arrival_time
            sim._admit_arrivals()
            continue
 
        sim.step(action_index=0, time_slice=quantum)
 
    return sim.get_metrics()
 
 
if __name__ == "__main__":
    from workload_generator import generate_workload
 
    workload = generate_workload(num_processes=5, seed=42)
    sim = CPUSimulator(workload)
 
    metrics = run_round_robin(sim, quantum=3)
    print("Round Robin metrics (quantum=3):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
 