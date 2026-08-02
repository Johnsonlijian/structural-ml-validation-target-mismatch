[CmdletBinding()]
param(
    [switch]$WhatIfAudit
)

<#
.SYNOPSIS
    Fail-closed, canonical-only scheduler for confirmatory protocol v3.2.

.DESCRIPTION
    This script is orchestration only. It reads protocol_v32.STAGE_ORDER once,
    launches exactly the two official shards (start=0 and start=50) for each
    stage, waits for both, and invokes the matching official aggregator before
    advancing. It never freezes, derives seeds, repairs, resumes, overwrites, or
    deletes scientific outputs.

    -WhatIfAudit performs only read-only command, dependency, freshness, and
    path-budget checks. It does not create freeze_v32, confirmatory_v32, a32,
    or pipeline_logs_v32.
#>

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:CvRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$script:ProjectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $script:CvRoot '..\..\..\..')
)
$script:PythonPath = Join-Path $script:ProjectRoot 'r30ds\Scripts\python.exe'
$script:ProtocolPath = Join-Path $script:CvRoot 'protocol_v32.py'
$script:CompilerRunner = Join-Path $script:CvRoot 'run_compiler_suite_v32.py'
$script:PredictiveRunner = Join-Path $script:CvRoot 'run_batch_v32.py'
$script:CompilerAggregator = Join-Path $script:CvRoot 'aggregate_compiler_suite_v32.py'
$script:PredictiveAggregator = Join-Path $script:CvRoot 'aggregate_batch_v32.py'
$script:FreezeRoot = Join-Path $script:CvRoot 'freeze_v32'
$script:ConfirmatoryRoot = Join-Path $script:CvRoot 'confirmatory_v32'
$script:AggregateRoot = Join-Path $script:CvRoot 'a32'
$script:LogRoot = Join-Path $script:CvRoot 'pipeline_logs_v32'
$script:PipelineLog = Join-Path $script:LogRoot 'pipeline.log'
$script:MaxWindowsPath = 260
$script:MaxConservativeCommandLine = 8191

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if (($Value.Length -gt 0) -and ($Value -notmatch '[\s"]')) {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes += 1
            continue
        }
        if ($character -eq '"') {
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * ($backslashes * 2)))
            }
            [void]$builder.Append('\"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Join-NativeArguments {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $quoted = @($Arguments | ForEach-Object { ConvertTo-NativeArgument -Value $_ })
    return [string]::Join(' ', $quoted)
}

function Format-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    return ((ConvertTo-NativeArgument -Value $Executable) + ' ' +
        (Join-NativeArguments -Arguments $Arguments))
}

function Invoke-NativeReadOnly {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = Join-NativeArguments -Arguments $Arguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables['PYTHONDONTWRITEBYTECODE'] = '1'

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw ("Could not start read-only dependency audit: {0}" -f $Executable)
    }
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $process.Dispose()

    return [pscustomobject]@{
        ExitCode = [int]$exitCode
        Stdout = [string]$stdout
        Stderr = [string]$stderr
    }
}

function Get-FrozenStageOrder {
    $code = "import json, protocol_v32; print(json.dumps(list(protocol_v32.STAGE_ORDER), separators=(',', ':')))"
    $result = Invoke-NativeReadOnly -Executable $script:PythonPath `
        -Arguments @('-B', '-c', $code) -WorkingDirectory $script:CvRoot
    if ($result.ExitCode -ne 0) {
        throw ("Could not read protocol_v32.STAGE_ORDER (exit={0}): {1}" -f
            $result.ExitCode, $result.Stderr.Trim())
    }
    try {
        $stages = [string[]](ConvertFrom-Json -InputObject $result.Stdout.Trim())
    }
    catch {
        throw ("protocol_v32.STAGE_ORDER did not produce valid JSON: {0}" -f
            $result.Stdout.Trim())
    }
    $stageCount = @($stages).Count
    $firstStage = if ($stageCount -gt 0) { [string]$stages[0] } else { '<empty>' }
    if (($stageCount -lt 2) -or ($firstStage -ne 'COMPILER_SEMANTIC')) {
        throw ("protocol_v32.STAGE_ORDER must begin with COMPILER_SEMANTIC and include predictive stages (count={0}, first={1})" -f
            $stageCount, $firstStage)
    }
    $unique = @($stages | Sort-Object -Unique)
    if ($unique.Count -ne $stages.Count) {
        throw 'protocol_v32.STAGE_ORDER contains duplicate stages'
    }
    foreach ($stage in $stages) {
        if (($stage -isnot [string]) -or ($stage -notmatch '^[A-Za-z0-9_.-]+$')) {
            throw ("Unsafe stage token read from protocol_v32.STAGE_ORDER: {0}" -f $stage)
        }
    }
    return [string[]]$stages
}

function Get-StageDirectoryName {
    param([Parameter(Mandatory = $true)][string]$Stage)
    if ($Stage -eq 'COMPILER_SEMANTIC') {
        return 'compiler_semantic'
    }
    return $Stage
}

function Get-RunnerArguments {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][ValidateSet(0, 50)][int]$Start
    )
    if ($Stage -eq 'COMPILER_SEMANTIC') {
        return [string[]]@('-B', $script:CompilerRunner, '--start', [string]$Start)
    }
    return [string[]]@(
        '-B', $script:PredictiveRunner, '--scenario', $Stage,
        '--start', [string]$Start
    )
}

function Get-AggregatorArguments {
    param([Parameter(Mandatory = $true)][string]$Stage)
    if ($Stage -eq 'COMPILER_SEMANTIC') {
        return [string[]]@('-B', $script:CompilerAggregator)
    }
    return [string[]]@('-B', $script:PredictiveAggregator, '--scenario', $Stage)
}

function Test-StaticDependencies {
    $required = @(
        $script:PythonPath,
        $script:ProtocolPath,
        $script:CompilerRunner,
        $script:PredictiveRunner,
        $script:CompilerAggregator,
        $script:PredictiveAggregator,
        (Join-Path $script:CvRoot 'run_compiler_suite.py'),
        (Join-Path $script:CvRoot 'run_batch.py'),
        (Join-Path $script:CvRoot 'aggregate_compiler_suite.py'),
        (Join-Path $script:CvRoot 'aggregate_batch.py')
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        throw ("Missing pipeline dependency:`n - " + ($missing -join "`n - "))
    }

    $pythonSources = @($required | Where-Object { $_.EndsWith('.py') })
    $astCode = @'
import ast
import pathlib
import sys
for item in sys.argv[1:]:
    ast.parse(pathlib.Path(item).read_text(encoding="utf-8"), filename=item)
print("AST_OK", len(sys.argv) - 1)
'@
    $astResult = Invoke-NativeReadOnly -Executable $script:PythonPath `
        -Arguments ([string[]](@('-B', '-c', $astCode) + $pythonSources)) `
        -WorkingDirectory $script:CvRoot
    if ($astResult.ExitCode -ne 0) {
        throw ("Python command dependency AST audit failed (exit={0}): {1}" -f
            $astResult.ExitCode, $astResult.Stderr.Trim())
    }

    $dependencyCode = @'
import importlib.metadata as metadata
import cvxpy
import datasail
import numpy
import pandas
import sklearn
assert metadata.version("datasail") == "1.3.0"
assert "SCIP" in cvxpy.installed_solvers()
print("R30DS_DEPENDENCIES_OK")
'@
    $dependencyResult = Invoke-NativeReadOnly -Executable $script:PythonPath `
        -Arguments @('-B', '-c', $dependencyCode) -WorkingDirectory $script:ProjectRoot
    if ($dependencyResult.ExitCode -ne 0) {
        throw ("r30ds dependency audit failed (exit={0}): {1}" -f
            $dependencyResult.ExitCode, $dependencyResult.Stderr.Trim())
    }

    return [pscustomobject]@{
        FilesChecked = $required.Count
        PythonAstChecked = $pythonSources.Count
        EnvironmentMessage = $dependencyResult.Stdout.Trim()
    }
}

function Test-PathAndCommandBudget {
    param([Parameter(Mandatory = $true)][string[]]$StageOrder)

    $pathCandidates = New-Object 'System.Collections.Generic.List[string]'
    $commandCandidates = New-Object 'System.Collections.Generic.List[string]'
    foreach ($stage in $StageOrder) {
        $stageDirectory = Get-StageDirectoryName -Stage $stage
        $resultDirectory = Join-Path $script:ConfirmatoryRoot $stageDirectory
        $aggregateDirectory = Join-Path $script:AggregateRoot $stageDirectory
        $stageLogDirectory = Join-Path $script:LogRoot $stage

        foreach ($candidate in @(
            (Join-Path $resultDirectory 'rep_0099.json'),
            (Join-Path $resultDirectory 'rep_0099.claim'),
            (Join-Path $resultDirectory 'worker_050_099.active'),
            (Join-Path $resultDirectory 'worker_050_099.done'),
            (Join-Path $resultDirectory '.__v32_tmp_4294967295'),
            (Join-Path $aggregateDirectory 'v32_aggregate_manifest.json'),
            (Join-Path $aggregateDirectory 'datasail_solver_status_summary.csv'),
            (Join-Path $stageLogDirectory 'w0.out'),
            (Join-Path $stageLogDirectory 'w0.err'),
            (Join-Path $stageLogDirectory 'w50.out'),
            (Join-Path $stageLogDirectory 'w50.err'),
            (Join-Path $stageLogDirectory 'agg.out'),
            (Join-Path $stageLogDirectory 'agg.err')
        )) {
            [void]$pathCandidates.Add([string]$candidate)
        }

        foreach ($start in @(0, 50)) {
            [void]$commandCandidates.Add((Format-NativeCommand `
                -Executable $script:PythonPath `
                -Arguments (Get-RunnerArguments -Stage $stage -Start $start)))
        }
        [void]$commandCandidates.Add((Format-NativeCommand `
            -Executable $script:PythonPath `
            -Arguments (Get-AggregatorArguments -Stage $stage)))
    }

    foreach ($candidate in @(
        (Join-Path $script:AggregateRoot '.__tmp_4294967295\datasail_solver_status_summary.csv'),
        (Join-Path $script:FreezeRoot 'environment_r30ds_pip_freeze.txt'),
        (Join-Path $script:FreezeRoot 'freeze_manifest.json'),
        (Join-Path $script:FreezeRoot 'freeze_manifest.sha256')
    )) {
        [void]$pathCandidates.Add([string]$candidate)
    }

    $longestPath = $pathCandidates | Sort-Object Length -Descending | Select-Object -First 1
    $longestCommand = $commandCandidates | Sort-Object Length -Descending | Select-Object -First 1
    if ($longestPath.Length -ge $script:MaxWindowsPath) {
        throw ("Path budget exceeded ({0} >= {1}): {2}" -f
            $longestPath.Length, $script:MaxWindowsPath, $longestPath)
    }
    if ($longestCommand.Length -gt $script:MaxConservativeCommandLine) {
        throw ("Conservative command-line budget exceeded ({0} > {1}): {2}" -f
            $longestCommand.Length, $script:MaxConservativeCommandLine,
            $longestCommand)
    }

    return [pscustomobject]@{
        LongestPathLength = [int]$longestPath.Length
        LongestPath = [string]$longestPath
        LongestCommandLength = [int]$longestCommand.Length
        LongestCommand = [string]$longestCommand
    }
}

function Get-ExecutionBlockers {
    param([switch]$IgnoreLogRoot)

    $blockers = New-Object 'System.Collections.Generic.List[string]'
    foreach ($requiredFreezeFile in @(
        (Join-Path $script:FreezeRoot 'freeze_manifest.json'),
        (Join-Path $script:FreezeRoot 'freeze_manifest.sha256'),
        (Join-Path $script:FreezeRoot 'confirmatory_seed_table.csv'),
        (Join-Path $script:FreezeRoot 'source_hashes.json'),
        (Join-Path $script:FreezeRoot 'excluded_pilot_registry.json')
    )) {
        if (-not (Test-Path -LiteralPath $requiredFreezeFile -PathType Leaf)) {
            [void]$blockers.Add(("missing final-freeze dependency (content not opened): {0}" -f
                $requiredFreezeFile))
        }
    }

    foreach ($pattern in @('.__f32_*', 'freeze_v32.__building_*')) {
        foreach ($entry in @(Get-ChildItem -LiteralPath $script:CvRoot -Force `
            -Filter $pattern -ErrorAction SilentlyContinue)) {
            [void]$blockers.Add(("stale freeze staging evidence: {0}" -f $entry.FullName))
        }
    }

    if (Test-Path -LiteralPath $script:FreezeRoot -PathType Container) {
        foreach ($entry in @(Get-ChildItem -LiteralPath $script:FreezeRoot -Force `
            -ErrorAction SilentlyContinue | Where-Object {
                $_.Name -like '.__freeze_v32_*' -or $_.Name -like '.__v32_tmp_*'
            })) {
            [void]$blockers.Add(("stale temporary inside final freeze: {0}" -f
                $entry.FullName))
        }
    }

    foreach ($root in @($script:ConfirmatoryRoot, $script:AggregateRoot)) {
        if (Test-Path -LiteralPath $root -PathType Container) {
            $entries = @(Get-ChildItem -LiteralPath $root -Force -ErrorAction Stop)
            if ($entries.Count -gt 0) {
                [void]$blockers.Add(("canonical root is not fresh ({0} entries): {1}" -f
                    $entries.Count, $root))
                foreach ($entry in @($entries | Select-Object -First 20)) {
                    [void]$blockers.Add(("existing canonical evidence: {0}" -f
                        $entry.FullName))
                }
            }
        }
    }

    if ((-not $IgnoreLogRoot) -and
        (Test-Path -LiteralPath $script:LogRoot -PathType Container)) {
        $logEntries = @(Get-ChildItem -LiteralPath $script:LogRoot -Force -ErrorAction Stop)
        if ($logEntries.Count -gt 0) {
            [void]$blockers.Add(("pipeline log root is not fresh ({0} entries): {1}" -f
                $logEntries.Count, $script:LogRoot))
            foreach ($entry in @($logEntries | Select-Object -First 20)) {
                [void]$blockers.Add(("existing log evidence: {0}" -f $entry.FullName))
            }
        }
    }

    return [string[]]$blockers
}

function Write-PipelineEvent {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = ('{0} {1}' -f ([DateTime]::UtcNow.ToString('o')), $Message)
    Add-Content -LiteralPath $script:PipelineLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Start-HiddenPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    if ((Test-Path -LiteralPath $StdoutPath) -or (Test-Path -LiteralPath $StderrPath)) {
        throw ("Refusing to overwrite process log: {0} or {1}" -f
            $StdoutPath, $StderrPath)
    }
    $argumentLine = Join-NativeArguments -Arguments $Arguments
    return Start-Process -FilePath $script:PythonPath -ArgumentList $argumentLine `
        -WorkingDirectory $script:CvRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
}

function Stop-LiveProcess {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process)
    try {
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction Stop
            $Process.WaitForExit()
        }
    }
    catch {
        Write-Warning ("Could not terminate peer process {0}: {1}" -f
            $Process.Id, $_.Exception.Message)
    }
}

function Wait-WorkerPair {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][object[]]$WorkerStates
    )

    while (@($WorkerStates | Where-Object { -not $_.Finished }).Count -gt 0) {
        foreach ($state in $WorkerStates) {
            if ((-not $state.Finished) -and $state.Process.HasExited) {
                $state.Process.WaitForExit()
                $state.Process.Refresh()
                $state.ExitCode = [int]$state.Process.ExitCode
                $state.Finished = $true
                Write-PipelineEvent -Message ("WORKER_EXIT stage={0} start={1} pid={2} exit={3}" -f
                    $Stage, $state.Start, $state.Process.Id, $state.ExitCode)
                if ($state.ExitCode -ne 0) {
                    foreach ($peer in $WorkerStates) {
                        if (-not $peer.Finished) {
                            Stop-LiveProcess -Process $peer.Process
                            $peer.Finished = $true
                            if ($peer.Process.HasExited) {
                                $peer.Process.Refresh()
                                $peer.ExitCode = [int]$peer.Process.ExitCode
                            }
                        }
                    }
                    throw ("Worker failure; pipeline stopped without recovery: stage={0}, start={1}, exit={2}, stdout={3}, stderr={4}" -f
                        $Stage, $state.Start, $state.ExitCode,
                        $state.StdoutPath, $state.StderrPath)
                }
            }
        }
        if (@($WorkerStates | Where-Object { -not $_.Finished }).Count -gt 0) {
            Start-Sleep -Milliseconds 1000
        }
    }
}

function Invoke-Stage {
    param([Parameter(Mandatory = $true)][string]$Stage)

    $stageDirectory = Get-StageDirectoryName -Stage $Stage
    $stageResultRoot = Join-Path $script:ConfirmatoryRoot $stageDirectory
    $stageAggregateRoot = Join-Path $script:AggregateRoot $stageDirectory
    if ((Test-Path -LiteralPath $stageResultRoot) -or
        (Test-Path -LiteralPath $stageAggregateRoot)) {
        throw ("Stage is not fresh; no resume is permitted: result={0}, aggregate={1}" -f
            $stageResultRoot, $stageAggregateRoot)
    }
    if (Test-Path -LiteralPath $script:AggregateRoot -PathType Container) {
        $aggregateTemps = @(Get-ChildItem -LiteralPath $script:AggregateRoot -Force `
            -Filter '.__tmp_*' -ErrorAction Stop)
        if ($aggregateTemps.Count -gt 0) {
            throw ("Stale aggregate temporary is a hard stop: {0}" -f
                $aggregateTemps[0].FullName)
        }
    }

    $stageLogRoot = Join-Path $script:LogRoot $Stage
    if (Test-Path -LiteralPath $stageLogRoot) {
        throw ("Stage log directory already exists; no overwrite is permitted: {0}" -f
            $stageLogRoot)
    }
    [void](New-Item -ItemType Directory -Path $stageLogRoot -ErrorAction Stop)
    Write-PipelineEvent -Message ("STAGE_START stage={0}" -f $Stage)

    $states = @()
    try {
        foreach ($start in @(0, 50)) {
            $tag = if ($start -eq 0) { 'w0' } else { 'w50' }
            $stdoutPath = Join-Path $stageLogRoot ($tag + '.out')
            $stderrPath = Join-Path $stageLogRoot ($tag + '.err')
            $arguments = Get-RunnerArguments -Stage $Stage -Start $start
            $process = Start-HiddenPython -Arguments $arguments `
                -StdoutPath $stdoutPath -StderrPath $stderrPath
            $states += [pscustomobject]@{
                Start = [int]$start
                Process = $process
                StdoutPath = $stdoutPath
                StderrPath = $stderrPath
                Finished = $false
                ExitCode = $null
            }
            Write-PipelineEvent -Message ("WORKER_START stage={0} start={1} pid={2}" -f
                $Stage, $start, $process.Id)
        }
        Wait-WorkerPair -Stage $Stage -WorkerStates $states
    }
    catch {
        foreach ($state in $states) {
            if (-not $state.Finished) {
                Stop-LiveProcess -Process $state.Process
            }
        }
        throw
    }

    $aggregateStdout = Join-Path $stageLogRoot 'agg.out'
    $aggregateStderr = Join-Path $stageLogRoot 'agg.err'
    $aggregateArguments = Get-AggregatorArguments -Stage $Stage
    $aggregateProcess = Start-HiddenPython -Arguments $aggregateArguments `
        -StdoutPath $aggregateStdout -StderrPath $aggregateStderr
    Write-PipelineEvent -Message ("AGGREGATOR_START stage={0} pid={1}" -f
        $Stage, $aggregateProcess.Id)
    $aggregateProcess.WaitForExit()
    $aggregateProcess.Refresh()
    $aggregateExit = [int]$aggregateProcess.ExitCode
    Write-PipelineEvent -Message ("AGGREGATOR_EXIT stage={0} pid={1} exit={2}" -f
        $Stage, $aggregateProcess.Id, $aggregateExit)
    if ($aggregateExit -ne 0) {
        throw ("Aggregator failure; pipeline stopped without recovery: stage={0}, exit={1}, stdout={2}, stderr={3}" -f
            $Stage, $aggregateExit, $aggregateStdout, $aggregateStderr)
    }

    $aggregateManifest = Join-Path $stageAggregateRoot 'v32_aggregate_manifest.json'
    if (-not (Test-Path -LiteralPath $aggregateManifest -PathType Leaf)) {
        throw ("Aggregator returned zero but final manifest is absent: {0}" -f
            $aggregateManifest)
    }
    Write-PipelineEvent -Message ("STAGE_COMPLETE stage={0} manifest={1}" -f
        $Stage, $aggregateManifest)
}

$dependencies = Test-StaticDependencies
$stageOrder = Get-FrozenStageOrder
$budget = Test-PathAndCommandBudget -StageOrder $stageOrder
$blockers = @(Get-ExecutionBlockers)

if ($WhatIfAudit) {
    Write-Host 'V32_PIPELINE_WHAT_IF_AUDIT'
    Write-Host ("cv_root={0}" -f $script:CvRoot)
    Write-Host ("python={0}" -f $script:PythonPath)
    Write-Host ("stage_order_source={0}:STAGE_ORDER" -f $script:ProtocolPath)
    Write-Host ("stages={0}; files_checked={1}; python_ast_checked={2}; environment={3}" -f
        $stageOrder.Count, $dependencies.FilesChecked,
        $dependencies.PythonAstChecked, $dependencies.EnvironmentMessage)
    Write-Host ("path_budget={0}/{1}; command_budget={2}/{3}" -f
        $budget.LongestPathLength, $script:MaxWindowsPath,
        $budget.LongestCommandLength, $script:MaxConservativeCommandLine)
    Write-Host ("longest_path={0}" -f $budget.LongestPath)
    for ($index = 0; $index -lt $stageOrder.Count; $index += 1) {
        $stage = $stageOrder[$index]
        Write-Host ("[{0:D2}] {1}" -f ($index + 1), $stage)
        Write-Host ("  worker0: {0}" -f (Format-NativeCommand `
            -Executable $script:PythonPath `
            -Arguments (Get-RunnerArguments -Stage $stage -Start 0)))
        Write-Host ("  worker50: {0}" -f (Format-NativeCommand `
            -Executable $script:PythonPath `
            -Arguments (Get-RunnerArguments -Stage $stage -Start 50)))
        Write-Host ("  aggregate: {0}" -f (Format-NativeCommand `
            -Executable $script:PythonPath `
            -Arguments (Get-AggregatorArguments -Stage $stage)))
    }
    if ($blockers.Count -gt 0) {
        Write-Host 'execution_ready=false'
        foreach ($blocker in $blockers) {
            Write-Host ("BLOCKER: {0}" -f $blocker)
        }
    }
    else {
        Write-Host 'execution_ready=true'
    }
    Write-Host 'WHAT_IF_ONLY: no freeze, seed-table read, canonical-root creation, runner, or aggregator was performed.'
    return
}

if ($blockers.Count -gt 0) {
    throw ("Fresh-run preflight failed; no recovery or overwrite will be attempted:`n - " +
        ($blockers -join "`n - "))
}

if (-not (Test-Path -LiteralPath $script:LogRoot)) {
    [void](New-Item -ItemType Directory -Path $script:LogRoot -ErrorAction Stop)
}
$lockPath = Join-Path $script:LogRoot '.pipeline.lock'
$lockStream = $null
$pipelineSucceeded = $false
try {
    $lockStream = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    $lockPayload = [System.Text.Encoding]::UTF8.GetBytes(
        ("pid={0}; started_utc={1}; stage_count={2}`n" -f
            $PID, [DateTime]::UtcNow.ToString('o'), $stageOrder.Count)
    )
    $lockStream.Write($lockPayload, 0, $lockPayload.Length)
    $lockStream.Flush()

    $raceBlockers = @(Get-ExecutionBlockers -IgnoreLogRoot)
    if ($raceBlockers.Count -gt 0) {
        throw ("Canonical roots changed after lock acquisition:`n - " +
            ($raceBlockers -join "`n - "))
    }

    Write-PipelineEvent -Message ("PIPELINE_START pid={0} stages={1}" -f
        $PID, $stageOrder.Count)
    foreach ($stage in $stageOrder) {
        Invoke-Stage -Stage $stage
    }
    Write-PipelineEvent -Message ("PIPELINE_COMPLETE stages={0}" -f $stageOrder.Count)
    $pipelineSucceeded = $true
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
    if ($pipelineSucceeded -and (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        Remove-Item -LiteralPath $lockPath -Force
    }
}
