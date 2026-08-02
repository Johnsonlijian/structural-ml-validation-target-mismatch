"""Official canonical-only compiler-conformance runner for protocol v3.2."""

from __future__ import annotations

import argparse

from protocol_v32 import (
    COMPILER_SCHEMA,
    PROTOCOL_ID,
    ProtocolPaths,
    atomic_create_json,
    begin_worker,
    bind_result,
    create_claim,
    expected_envelope,
    finish_worker,
    runtime_preflight,
    validate_shard,
    activate_base_runner,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one immutable v3.2 compiler shard in its canonical root."
    )
    parser.add_argument("--start", type=int, required=True, choices=(0, 50))
    args = parser.parse_args()
    start, count = int(args.start), 50
    validate_shard(start, count)

    paths = ProtocolPaths.canonical()
    context = runtime_preflight(paths)
    scenario = "COMPILER_SEMANTIC"
    worker_state = begin_worker(context, scenario, start, count)
    if worker_state == "skip_done":
        print(f"SKIP_DONE protocol={PROTOCOL_ID} scenario={scenario} start={start}")
        return

    activate_base_runner(context)
    import run_compiler_suite as suite

    for replicate in range(start, start + count):
        refreshed = runtime_preflight(paths)
        if refreshed.manifest_sha256 != context.manifest_sha256:
            raise RuntimeError("v3.2 freeze changed during compiler execution")
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
            payload = suite.run_suite_replicate(replicate)
            postflight = runtime_preflight(paths)
            if postflight.manifest_sha256 != context.manifest_sha256:
                raise RuntimeError("v3.2 freeze changed during compiler replicate")
            if payload.get("status") != "complete":
                raise RuntimeError("compiler suite returned a non-complete record")
            bound = bind_result(
                payload,
                expected_envelope(context, scenario, replicate),
                claim_hash,
            )
        except Exception as exc:
            failed = True
            failure = {
                "schema_version": COMPILER_SCHEMA,
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
