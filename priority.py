
from simulator import CPUSimulator
 
 
def run_priority(sim: CPUSimulator) -> dict:
    """
    Run the Priority scheduling policy on a (freshly reset) simulator
    until all processes finish, then return the performance metrics.
    """
    sim.reset()
 
    while not sim.is_done():
        if not sim.ready_queue:
            sim.time = sim._pending[0].arrival_time
            sim._admit_arrivals()
            continue
 
        # Find the index of the process with the lowest priority
        # number (i.e. the highest actual priority) in the ready queue.
        best_index = min(
            range(len(sim.ready_queue)),
            key=lambda i: sim.ready_queue[i].priority,
        )
 
        sim.step(action_index=best_index, time_slice=None)
 
    return sim.get_metrics()
 
 
if __name__ == "__main__":
    from workload_generator import generate_workload
 
    workload = generate_workload(num_processes=5, seed=42)
    sim = CPUSimulator(workload)
 
    metrics = run_priority(sim)
    print("Priority scheduling metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")