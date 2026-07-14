Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "E:\Project\python\Axon_v2.6Exp"
$PythonExecutable = "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe"
$Supervisor = "E:\Project\python\Axon_v2.6Exp\scripts\run_loop166_phase_b1_step4096_recovery_v2_supervisor.py"
$LaunchReceipt = "E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop166\phase_b1_step4096_recovery_v2_launch_receipt.json"
$ExitReceipt = "E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop166\phase_b1_step4096_recovery_v2_exit_receipt.json"
$StdoutLog = "E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop166\phase_b1_step4096_recovery_v2_stdout.log"
$StderrLog = "E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop166\phase_b1_step4096_recovery_v2_stderr.log"

foreach ($SourcePath in @($ProjectRoot, $PythonExecutable, $Supervisor)) {
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Required detached-launch source is unavailable: $SourcePath"
    }
    $SourceItem = Get-Item -LiteralPath $SourcePath
    if (($SourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Detached-launch source cannot be a reparse point: $SourcePath"
    }
}

foreach ($OutputPath in @($LaunchReceipt, $ExitReceipt, $StdoutLog, $StderrLog)) {
    if (Test-Path -LiteralPath $OutputPath) {
        throw "Detached-launch output already exists: $OutputPath"
    }
}

$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $PythonExecutable
$StartInfo.Arguments = "-u `"$Supervisor`""
$StartInfo.WorkingDirectory = $ProjectRoot
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true

foreach ($EnvironmentKey in @($StartInfo.EnvironmentVariables.Keys)) {
    if ($EnvironmentKey.StartsWith("AXON_B1_RECOVERY_V2_", [System.StringComparison]::OrdinalIgnoreCase)) {
        $StartInfo.EnvironmentVariables.Remove($EnvironmentKey)
    }
}
$StartInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1"

$SupervisorProcess = [System.Diagnostics.Process]::Start($StartInfo)
if ($null -eq $SupervisorProcess) {
    throw "Detached recovery supervisor did not start"
}

$ReceiptDeadline = [DateTime]::UtcNow.AddSeconds(30)
while (-not (Test-Path -LiteralPath $LaunchReceipt)) {
    if ($SupervisorProcess.HasExited) {
        throw "Detached recovery supervisor exited before persisting its launch receipt"
    }
    if ([DateTime]::UtcNow -ge $ReceiptDeadline) {
        throw "Detached recovery supervisor did not persist its launch receipt in time"
    }
    Start-Sleep -Milliseconds 100
}

$LaunchPayload = Get-Content -LiteralPath $LaunchReceipt -Raw | ConvertFrom-Json
if (
    $LaunchPayload.schema -ne "axon_loop166_phase_b1_step4096_recovery_v2_supervisor_launch_v1" -or
    $LaunchPayload.status -ne "supervisor_launch_frozen_before_controller_start" -or
    [int64]$LaunchPayload.supervisor_pid -le 0
) {
    throw "Detached recovery supervisor launch receipt is invalid"
}

[pscustomobject]@{
    schema = "axon_loop166_phase_b1_step4096_recovery_v2_detached_launch_v1"
    status = "detached_supervisor_started_and_launch_receipt_observed"
    launcher_pid = $SupervisorProcess.Id
    supervisor_pid = [int64]$LaunchPayload.supervisor_pid
    working_directory = $ProjectRoot
    python_executable = $PythonExecutable
    launch_receipt = $LaunchReceipt
    exit_receipt = $ExitReceipt
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    python_unbuffered = $true
} | ConvertTo-Json -Depth 3
