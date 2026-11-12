# Trains roles one after another (never in parallel), optionally evaluating each.
#
#   .\train_roles.ps1                         # every role in roles/kinds.json
#   .\train_roles.ps1 -Roles block,arms       # just these
#   .\train_roles.ps1 -Roles name -Eval       # train, then eval_role (greedy + swap into the oracle swarm)
#   .\train_roles.ps1 -Roles pat -Extra '--epochs','5','--batch','32'
#
# Per-role settings (epochs, batch, lr, own model shape) come from roles/kinds.json;
# anything in -Extra is passed to train_role and wins over the file.
param(
    [string[]]$Roles = @(),
    [string]$RoleFile = "roles/kinds.json",
    [switch]$Eval,
    [switch]$Resume,
    [string[]]$Extra = @()
)
# cargo writes warnings to stderr, which PowerShell 5.1 would turn into errors; check exit codes instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if ($Roles.Count -eq 0) {
    $Roles = (Get-Content $RoleFile -Raw | ConvertFrom-Json).roles | Where-Object { -not $_.written } | ForEach-Object { $_.name }
}
foreach ($r in $Roles) {
    Write-Host "=== train $r ===" -ForegroundColor Cyan
    $cmd = @("--roles", $RoleFile, "--role", $r) + $Extra
    if ($Resume) { $cmd += "--resume" }
    cargo run --release --bin train_role -- @cmd
    if ($LASTEXITCODE -ne 0) { throw "train_role $r failed" }
    if ($Eval) {
        Write-Host "=== eval $r ===" -ForegroundColor Cyan
        cargo run --release --bin eval_role -- --roles $RoleFile --role $r
        if ($LASTEXITCODE -ne 0) { throw "eval_role $r failed" }
    }
}
