"""
CPU Scheduling Simulator
-------------------------
This is the core engine. It does NOT decide which process to run —
that decision is made by whichever scheduler (FCFS, Round Robin,
Priority, or later the RL agent) is plugged in from outside.

The simulator's job is only to:
  - Track simulated time
  - Admit processes into the ready queue as they arrive
  - Run whichever process the scheduler picks
  - Update waiting time / remaining burst time / completion time
  - Count context switches
  - Report final performance metrics

Non-preemptive by default: once a process is picked, it runs until
either it finishes or a given time_slice runs out (time_slice is used
for Round Robin; leave it as None for FCFS/Priority to run to completion).
"""

"""
CPU Scheduling Simulator
-------------------------
"""

from dataclasses import dataclass, field
from workload_generator import Process


@dataclass
class SimProcess:
    process: Process
    remaining_burst_time: int = field(init=False)
    waiting_time: int = 0
    completion_time: int = None
    turnaround_time: int = None

    def __post_init__(self):
        self.remaining_burst_time = self.process.burst_time

    @property
    def id(self):
        return self.process.id

    @property
    def arrival_time(self):
        return self.process.arrival_time

    @property
    def priority(self):
        return self.process.priority


class CPUSimulator:
    def __init__(self, processes: list):
        self._original_processes = processes
        self.reset()

    def reset(self):
        self.time = 0
        self.ready_queue = []
        self.completed = []
        self.context_switches = 0
        self._last_run_id = None

        self._pending = sorted(
            (SimProcess(process=p) for p in self._original_processes),
            key=lambda sp: sp.arrival_time,
        )

        self._admit_arrivals()
        return self.get_ready_queue_snapshot()

    def _admit_arrivals(self):
        while self._pending and self._pending[0].arrival_time <= self.time:
            self.ready_queue.append(self._pending.pop(0))

    def is_done(self) -> bool:
        return not self.ready_queue and not self._pending

    def get_ready_queue_snapshot(self):
        return list(self.ready_queue)

    def step(self, action_index: int, time_slice: int = None):
        if not self.ready_queue:
            raise ValueError("step() called with an empty ready queue.")

        if not (0 <= action_index < len(self.ready_queue)):
            raise IndexError(
                f"action_index {action_index} out of range for ready "
                f"queue of length {len(self.ready_queue)}"
            )

        proc = self.ready_queue.pop(action_index)

        context_switch = proc.id != self._last_run_id
        if context_switch:
            self.context_switches += 1
        self._last_run_id = proc.id

        run_time = proc.remaining_burst_time
        if time_slice is not None:
            run_time = min(run_time, time_slice)

        for other in self.ready_queue:
            other.waiting_time += run_time

        self.time += run_time
        proc.remaining_burst_time -= run_time

        self._admit_arrivals()

        finished = proc.remaining_burst_time <= 0
        if finished:
            proc.completion_time = self.time
            proc.turnaround_time = proc.completion_time - proc.arrival_time
            self.completed.append(proc)
        else:
            self.ready_queue.append(proc)

        done = self.is_done()

        info = {
            "ran_process_id": proc.id,
            "run_time": run_time,
            "context_switch": context_switch,
            "finished": finished,
            "current_time": self.time,
        }

        return self.get_ready_queue_snapshot(), done, info

    def get_metrics(self) -> dict:
        if not self.completed:
            return {
                "avg_waiting_time": 0.0,
                "avg_turnaround_time": 0.0,
                "context_switches": self.context_switches,
                "fairness_index": 0.0,
            }

        waiting_times = [p.waiting_time for p in self.completed]
        turnaround_times = [p.turnaround_time for p in self.completed]

        avg_waiting = sum(waiting_times) / len(waiting_times)
        avg_turnaround = sum(turnaround_times) / len(turnaround_times)

        n = len(waiting_times)
        sum_x = sum(waiting_times)
        sum_x2 = sum(x ** 2 for x in waiting_times)
        fairness = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0

        return {
            "avg_waiting_time": avg_waiting,
            "avg_turnaround_time": avg_turnaround,
            "context_switches": self.context_switches,
            "fairness_index": fairness,
        }


if __name__ == "__main__":
    # Run all three classical schedulers on the SAME workload, so the
    # comparison is fair -- every algorithm faces identical processes.
    from workload_generator import generate_workload
    from fcfs import run_fcfs
    from round_robin import run_round_robin
    from priority import run_priority

    workload = generate_workload(num_processes=10, seed=42)

    print(f"Workload (seed=42, 5 processes):")
    for p in workload:
        print(f"  {p}")
    print()

    results = {
        "FCFS": run_fcfs(CPUSimulator(workload)),
        "Round Robin": run_round_robin(CPUSimulator(workload), quantum=3),
        "Priority": run_priority(CPUSimulator(workload)),
    }

    metric_names = ["avg_waiting_time", "avg_turnaround_time",
                     "context_switches", "fairness_index"]

    header = f"{'Metric':<22}" + "".join(f"{name:<15}" for name in results)
    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = f"{metric:<22}"
        for scheduler_name in results:
            value = results[scheduler_name][metric]
            row += f"{value:<15.3f}"
        print(row)