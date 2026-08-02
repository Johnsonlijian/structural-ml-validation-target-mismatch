"""Official canonical-only predictive runner for protocol v3.2."""

from __future__ import annotations

import argparse

from protocol_v32 import (
    PREDICTIVE_SCHEMA,
    PROTOCOL_ID,
    ProtocolPaths,
    SCENARIOS,
    activate_base_runner,
    atomic_create_json,
    begin_worker,
    bind_result,
    create_claim,
    expected_envelope,
    finish_worker,
    runtime_preflight,
    validate_scenario_name,
    validate_shard,
)


PREDICTIVE_SCENARIOS = tuple(sorted(SCENARIOS.difference({"COMPILER_SEMANTIC"})))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one immutable v3.2 predictive shard in its canonical root."
    )
    parser.add_argument("--scenario", required=True, choices=PREDICTIVE_SCENARIOS)
    parser.add_argument("--start", type=int, required=True, choices=(0, 50))
    args = parser.parse_args()
    scenario, start, count = str(args.scenario), int(args.start), 50
    validate_scenario_name(scenario)
    validate_shard(start, count)

    paths = ProtocolPaths.canonical()
    context = runtime_preflight(paths)
    worker_state = begin_worker(context, scenario, start, count)
    if worker_state == "skip_done":
        print(f"SKIP_DONE protocol={PROTOCOL_ID} scenario={scenario} start={start}")
        return
    runner = activate_base_runner(context)

    for replicate in range(start, start + count):
        refreshed = runtime_preflight(paths)
        if refreshed.manifest_sha256 != context.manifest_sha256:
            raise RuntimeError("v3.2 freeze changed during predictive execution")
        result_path = paths.result_path(scenario, replicate)
        _, claim_hash = create_claim(
            context,
            scenario,
            replicate,
            start=start,
            count=count,
        )
        failed = False
        try:
            payload = runner.run_replicate(
                scenario=scenario,
                replicate=replicate,
                skip_datasail=False,
                datasail_max_sec=60,
            )
            postflight = runtime_preflight(paths)
            if postflight.manifest_sha256 != context.manifest_sha256:
                raise RuntimeError("v3.2 freeze changed during predictive replicate")
            if payload.get("status") != "complete":
                raise RuntimeError("predictive runner returned a non-complete record")
            bound = bind_result(
                payload,
                expected_envelope(context, scenario, replicate),
                claim_hash,
            )
        except Exception as exc:
            failed = True
            failure = {
                "schema_version": PREDICTIVE_SCHEMA,
                "scenario": scenario,
                "replicate": int(replicate),
                "seeds": context.row(scenario, replicate).seeds,
                "seed_table_sha256": context.seed_table_sha256,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            bound = bind_result(
                failure,
                expected_envelope(context, scenario, replicate),
                claim_hash,
            )
        atomic_create_json(result_path, bound)
        print(
            f"{bound['status'].upper()} protocol={PROTOCOL_ID} "
            f"scenario={scenario} replicate={replicate}"
        )
        if failed:
            raise SystemExit(1)
    runtime_preflight(paths)
    finish_worker(context, scenario, start, count)


if __name__ == "__main__":
    main()
