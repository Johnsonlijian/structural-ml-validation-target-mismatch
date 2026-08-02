# Protocol v3.2 pipeline controller

`run_pipeline_v32.ps1` is a fail-closed Windows PowerShell 5.1 controller for
the already-defined v3.2 scientific protocol. It does not define scenarios,
derive seeds, freeze inputs, repair partial runs, or modify scientific code.

Read-only audit (safe before the final freeze):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_v32.ps1 -WhatIfAudit
```

The audit reads `protocol_v32.STAGE_ORDER`, checks the project-local `r30ds`
interpreter and command dependencies, constructs every official command, and
checks conservative Windows path and command-line budgets. It only reports
whether the later execution would be ready; it does not open the seed table or
create `freeze_v32`, `confirmatory_v32`, `a32`, or `pipeline_logs_v32`.

Execution is intentionally single-shot and is not authorized by this file.
After a separately authorized final freeze, invoke the same command without
`-WhatIfAudit`.
Every stage launches only the official `start=0` and `start=50` runners, waits
for both exit codes, and then launches the official aggregator. The next stage
is unreachable unless all three processes return zero and the final aggregate
manifest exists.

Logs are written outside the canonical scientific roots under
`pipeline_logs_v32/<stage>/`. Any pre-existing canonical result/aggregate
entry, active/temp evidence, stale freeze staging, or non-empty pipeline log
root is a hard stop. The controller never infers a resume point and never
deletes or overwrites evidence.
