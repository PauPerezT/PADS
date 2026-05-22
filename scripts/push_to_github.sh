#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Initialize this folder as a git repo, set up Git LFS for the checkpoint
# files, and push to a new branch on PauPerezT/PADS.
#
# Run from the PADS-optimized/ folder with Git Bash (Windows) or any shell
# with git + git-lfs installed.
#
# Usage:
#   bash scripts/push_to_github.sh                      # pushes to branch 'optimized'
#   BRANCH=my-branch bash scripts/push_to_github.sh     # custom branch name
#   REMOTE=git@github.com:PauPerezT/PADS.git bash scripts/push_to_github.sh
#   ORIG_REPO=../PADS bash scripts/push_to_github.sh    # path to your local PADS clone (to grab checkpoints)
# -----------------------------------------------------------------------------
set -euo pipefail

BRANCH="${BRANCH:-optimized}"
REMOTE="${REMOTE:-https://github.com/PauPerezT/PADS.git}"
ORIG_REPO="${ORIG_REPO:-}"

# Move to repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo ">>> Working in: $(pwd)"
echo ">>> Remote   : ${REMOTE}"
echo ">>> Branch   : ${BRANCH}"
echo

# 1. Pre-flight: make sure git + git-lfs are installed.
command -v git >/dev/null   || { echo "ERROR: git not installed"; exit 1; }
command -v git-lfs >/dev/null || { echo "ERROR: git-lfs not installed. See https://git-lfs.com"; exit 1; }

# 2. Remove any half-built .git folder (the sandbox may have left one behind).
if [ -d .git ] && [ ! -f .git/HEAD ]; then
  echo ">>> Removing broken .git directory"
  chmod -R u+w .git 2>/dev/null || true
  rm -rf .git
fi

# 3. git init (idempotent: skipped if already a repo).
if [ ! -d .git ]; then
  git init -b main
  git config user.email "$(git config --global user.email || echo paula.andrea.perez@fau.de)"
  git config user.name  "$(git config --global user.name  || echo 'Paula Perez-Toro')"
fi

# 4. Git LFS setup.
git lfs install
git lfs track "*.ckpt" "*.ckp"
# .gitattributes is created by git-lfs track; stage it now.

# 5. Optional: copy the checkpoint files from a local PADS clone so we have
#    something to commit with LFS. Skip if they're already present here.
need_arousal="checkpoints/one_sec/arousal_checkpoint.ckpt"
if [ ! -f "$need_arousal" ]; then
  if [ -n "$ORIG_REPO" ] && [ -d "$ORIG_REPO/checkpoints" ]; then
    echo ">>> Copying checkpoints from $ORIG_REPO/checkpoints"
    mkdir -p checkpoints/one_sec
    cp -n "$ORIG_REPO/checkpoints/"*.ckp           checkpoints/         2>/dev/null || true
    cp -n "$ORIG_REPO/checkpoints/one_sec/"*.ckpt  checkpoints/one_sec/ 2>/dev/null || true
  else
    echo ">>> NOTE: no checkpoints found in checkpoints/ and ORIG_REPO not set."
    echo "          The commit will only contain code + small assets."
    echo "          To include trained checkpoints, re-run with:"
    echo "            ORIG_REPO=/path/to/your/PADS bash scripts/push_to_github.sh"
  fi
fi

# 6. Stage and commit everything.
git add .
if git diff --cached --quiet; then
  echo ">>> Nothing new to commit"
else
  git commit -m "Optimized PADS: clean library, Streamlit web UI, ~2.3x speedup

- Modular pip-installable 'pads' package (features, models, inference, viz)
- Vectorised mel-spectrogram extraction (sliding_window_view + cached filterbank)
- Lazy torch import: feature extraction works without torch
- Streamlit web app replaces the PySide6 desktop dashboard
- Numerically identical output to the original (max drift <3e-6)
- 9/9 pytest smoke tests passing
- Polished README, examples, Makefile, pyproject.toml"
fi

# 7. Configure remote and branch.
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

git checkout -B "$BRANCH"

# 8. Push.
echo
echo ">>> About to push branch '${BRANCH}' to ${REMOTE}"
echo "    You'll be prompted for GitHub credentials (PAT recommended)."
echo
read -r -p "Continue? [y/N] " ok
case "$ok" in
  y|Y|yes|YES) ;;
  *) echo "Aborted - nothing pushed. Run 'git push -u origin ${BRANCH}' manually when ready."; exit 0 ;;
esac

git push -u origin "$BRANCH"

echo
echo ">>> Done. Branch '${BRANCH}' pushed to ${REMOTE}"
echo ">>> Open a PR at: https://github.com/PauPerezT/PADS/compare/${BRANCH}?expand=1"
