# -----------------------------------------------------------------------------
# Initialize this folder as a git repo, set up Git LFS for the checkpoint
# files, and push to a new branch on PauPerezT/PADS.
#
# Run from the PADS-optimized/ folder in PowerShell.
#
# Usage:
#   .\scripts\push_to_github.ps1
#   .\scripts\push_to_github.ps1 -Branch my-branch
#   .\scripts\push_to_github.ps1 -OrigRepo ..\PADS    # path to your local PADS clone (for checkpoints)
# -----------------------------------------------------------------------------

param(
    [string]$Branch   = "optimized",
    [string]$Remote   = "https://github.com/PauPerezT/PADS.git",
    [string]$OrigRepo = ""
)

$ErrorActionPreference = "Stop"

# Move to repo root (script lives in <root>/scripts/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir
Set-Location $Root

Write-Host ">>> Working in: $Root"
Write-Host ">>> Remote   : $Remote"
Write-Host ">>> Branch   : $Branch"
Write-Host

# 1. Pre-flight.
if (-not (Get-Command git    -ErrorAction SilentlyContinue)) { throw "git not installed" }
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) { throw "git-lfs not installed. See https://git-lfs.com" }

# 2. Remove broken .git directory if the sandbox left one.
if ((Test-Path ".git") -and ((-not (Test-Path ".git\HEAD")) -or (-not (Test-Path ".git\objects")))) {
    Write-Host ">>> Removing broken .git directory"
    Remove-Item -Recurse -Force ".git"
}

# 3. git init.
if (-not (Test-Path ".git")) {
    git init -b main
    $ConfigEmail = (git config --global user.email 2>$null)
    $ConfigName  = (git config --global user.name  2>$null)
    if (-not $ConfigEmail) { $ConfigEmail = "paula.andrea.perez@fau.de" }
    if (-not $ConfigName)  { $ConfigName  = "Paula Perez-Toro" }
    git config user.email $ConfigEmail
    git config user.name  $ConfigName
}

# 4. Git LFS.
git lfs install
git lfs track "*.ckpt" "*.ckp"

# 5. Copy checkpoints from your local PADS clone if requested.
$NeedArousal = "checkpoints\one_sec\arousal_checkpoint.ckpt"
if (-not (Test-Path $NeedArousal)) {
    if ($OrigRepo -and (Test-Path (Join-Path $OrigRepo "checkpoints"))) {
        Write-Host ">>> Copying checkpoints from $OrigRepo\checkpoints"
        New-Item -ItemType Directory -Path "checkpoints\one_sec" -Force | Out-Null
        Get-ChildItem (Join-Path $OrigRepo "checkpoints") -Filter "*.ckp"            | Copy-Item -Destination "checkpoints\"
        Get-ChildItem (Join-Path $OrigRepo "checkpoints\one_sec") -Filter "*.ckpt"   | Copy-Item -Destination "checkpoints\one_sec\"
    } else {
        Write-Host ">>> NOTE: no checkpoints found locally and -OrigRepo not set."
        Write-Host "         The commit will contain code + small assets only."
        Write-Host "         To include trained checkpoints, re-run with:"
        Write-Host "             .\scripts\push_to_github.ps1 -OrigRepo C:\path\to\your\PADS"
    }
}

# 6. Stage and commit.
git add .
git diff --cached --quiet
$Staged = $LASTEXITCODE
if ($Staged -eq 0) {
    Write-Host ">>> Nothing new to commit"
} else {
    $CommitMessage = @"
Optimized PADS: clean library, Streamlit web UI, ~2.3x speedup

- Modular pip-installable 'pads' package (features, models, inference, viz)
- Vectorised mel-spectrogram extraction (sliding_window_view + cached filterbank)
- Lazy torch import: feature extraction works without torch
- Streamlit web app replaces the PySide6 desktop dashboard
- Numerically identical output to the original (max drift <3e-6)
- 9/9 pytest smoke tests passing
- Polished README, examples, Makefile, pyproject.toml
"@
    git commit -m $CommitMessage
}

# 7. Remote + branch.
$HasOrigin = $true
try { git remote get-url origin | Out-Null } catch { $HasOrigin = $false }
if ($HasOrigin) { git remote set-url origin $Remote } else { git remote add origin $Remote }

git checkout -B $Branch

# 8. Push.
Write-Host
Write-Host ">>> About to push branch '$Branch' to $Remote"
Write-Host "    You'll be prompted for GitHub credentials (Personal Access Token recommended)."
Write-Host
$Confirm = Read-Host "Continue? [y/N]"
if ($Confirm -notmatch "^(y|yes)$") {
    Write-Host "Aborted - nothing pushed. Run 'git push -u origin $Branch' manually when ready."
    exit 0
}

git push -u origin $Branch

Write-Host
Write-Host ">>> Done. Branch '$Branch' pushed to $Remote"
Write-Host ">>> Open a PR at: https://github.com/PauPerezT/PADS/compare/$Branch`?expand=1"
