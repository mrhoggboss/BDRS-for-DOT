"""PDOT's warm Julia process.

One Julia process serves every PDOT solve of an orchestrator run, so the
Julia/CUDA JIT is paid once. Before each measured solve the driver
(``vendor/PDOT.jl/bench_driver.jl``) resets the process state and warms up on
the instance.
"""

from pathlib import Path

from src.benchmark.dispatch import (
    _make_synthesized_record, _stamp_regime, _stamp_time_limit,
)
from src.protocol.instance_record import InstanceSpec, ResultRecord
from src.runners.run_pdot import JuliaPdot, pdot_size_gate


class PdotAdapter:
    """One ``JuliaPdot``, started and warmed up at construction. A row the
    driver does not write is built by ``_make_synthesized_record``; every row
    records ``spec.time_limit_s``."""

    def __init__(self, driver_path: Path, project_dir: Path, *,
                 startup_timeout_s: float = 300.0):
        if not driver_path.is_file():
            raise FileNotFoundError(f"PDOT bench driver not found: {driver_path}")
        if not (project_dir / "Project.toml").is_file():
            raise FileNotFoundError(f"PDOT project not found at {project_dir}")
        self.julia = JuliaPdot(driver_path, project_dir, startup_timeout_s=startup_timeout_s)
        self.julia.launch()

    def run(self, spec: InstanceSpec) -> ResultRecord:
        # Gated before the cost is materialised: a gated cost is 32 GiB or more.
        _gate = pdot_size_gate(spec)
        if _gate is not None:
            return _stamp_time_limit(_make_synthesized_record(
                spec, "PDOT", status="oom",
                error_kind="pdot_device_oom_structural", error_msg=_gate,
            ), spec)
        # PDOT reads its cost from a file, so a procedural cost is written for
        # this solve and removed after it. Imported here: dense_cost needs
        # torch, and the orchestrator must import without it.
        from src.instances.dense_cost import materialized_cost_path
        with materialized_cost_path(spec) as spec_with_cost:
            rec = self.julia.solve(spec_with_cost, failure=lambda **kw: _make_synthesized_record(
                spec_with_cost, "PDOT", **kw))
        return _stamp_time_limit(_stamp_regime(rec, "PDOT"), spec)

    def shutdown(self) -> None:
        self.julia.shutdown()
