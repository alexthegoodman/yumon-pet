# The whole swarm RL pipeline, one step after another (never in parallel):
#   1 baseline  run_swarm on the current weights  -> out/phase3 (kept for comparison)
#   2 negatives gen_negatives, hole then program   -> out/critic/neg_*.jsonl
#   3 critics   train_critic, hole then program    -> ../checkpoints/swarm/critic_*
#   4 rl        train_rl                           -> ../checkpoints/swarm-rl
#   5 eval      run_swarm on the RL weights        -> out/phase6
#
#   .\train_rl_all.ps1                                  # everything
#   .\train_rl_all.ps1 -Skip baseline,negatives         # skip steps (names above)
#   .\train_rl_all.ps1 -Only rl,eval -Resume            # just these; -Resume continues RL
#   .\train_rl_all.ps1 -HoleNegatives 5000 -Iters 50    # smaller run
#   .\train_rl_all.ps1 -CriticExtra '--init','<checkpoint>'   # pretrained critic instead of scratch
#
# Stops at the first failing step. Read the critic accuracy before letting RL run: if the
# swarm-negative accuracy is near 50% the reward means nothing (use -Skip critics,rl and look).
param(
    [string[]]$Skip = @(),
    [string[]]$Only = @(),
    [int]$HoleNegatives = 20000,
    [int]$ProgramNegatives = 3000,
    [int]$CriticEpochs = 1,
    [int]$Iters = 100,
    [int]$Docs = 8,
    [int]$Group = 4,
    [int]$HeldOut = 50,
    [switch]$Resume,
    [switch]$NoCheck,
    [string[]]$CriticExtra = @(),
    [string[]]$RlExtra = @()
)
# cargo writes warnings to stderr, which PowerShell 5.1 would turn into errors; check exit codes instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Want($step) {
    if ($Only.Count -gt 0) { return $Only -contains $step }
    return -not ($Skip -contains $step)
}
function Run($title, [string[]]$cargoArgs) {
    Write-Host "=== $title ===" -ForegroundColor Cyan
    $t = Get-Date
    cargo run --release @cargoArgs
    if ($LASTEXITCODE -ne 0) { throw "$title failed" }
    Write-Host ("    {0:N0}s" -f ((Get-Date) - $t).TotalSeconds) -ForegroundColor DarkGray
}

$ck = "../checkpoints/swarm"
$neg = "out/critic"
$extraCheck = @(); if ($NoCheck) { $extraCheck = @("--no-check") }

if (Want "baseline") {
    Run "baseline swarm on held-out docs" (@("--bin", "run_swarm", "--", "--held-out", $HeldOut, "--out", "out/phase3") + $extraCheck)
}
if (Want "negatives") {
    Run "negatives (hole)" @("--bin", "gen_negatives", "--", "--level", "hole", "--n", $HoleNegatives, "--out", "$neg/neg_hole.jsonl")
    Run "negatives (program)" @("--bin", "gen_negatives", "--", "--level", "program", "--n", $ProgramNegatives, "--out", "$neg/neg_program.jsonl")
}
if (Want "critics") {
    foreach ($level in "hole", "program") {
        Run "critic ($level)" (@("--bin", "train_critic", "--", "--steps", "1024", "--limit-chunks", "1024", "--val-items", "32", "--level", $level, "--negatives", "$neg/neg_$level.jsonl",
            "--epochs", $CriticEpochs, "--out", "$ck/critic_$level") + $CriticExtra)
    }
}
if (Want "rl") {
    $rl = @("--bin", "train_rl", "--", "--critic-hole", "$ck/critic_hole", "--critic-program", "$ck/critic_program",
        "--iters", $Iters, "--docs", $Docs, "--group", $Group) + $extraCheck + $RlExtra
    if ($Resume) { $rl += "--resume" }
    Run "RL" $rl
}
if (Want "eval") {
    Run "swarm after RL on held-out docs" (@("--bin", "run_swarm", "--", "--held-out", $HeldOut, "--out", "out/phase6",
        "--planner", "../checkpoints/swarm-rl/planner", "--workers", "../checkpoints/swarm-rl") + $extraCheck)
    Write-Host "Compare out/phase3/report.json (before) with out/phase6/report.json (after)." -ForegroundColor Green
}
