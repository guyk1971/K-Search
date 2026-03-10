# Layered Docker Environment for Claude Code

This document describes the layered Docker approach for running Claude Code across multiple projects with a single authenticated base image.

## Overview

Instead of authenticating Claude Code separately for each project, this approach uses:

1. **Base Image** (`nvcr-pytorch:25.09-py3-claude-authenticated`): Contains PyTorch + Claude Code CLI + keyring support + OAuth credentials
2. **Project Images** (e.g., `my_envtools:latest` or `your-project:latest`): Built on top of the authenticated base, adds project-specific dependencies (from `requirements.txt`, `pyproject.toml`, or uv when `uv.lock` is present)

```
┌─────────────────────────────────────┐
│     your-project:latest             │  ← Project-specific deps (from repo root)
├─────────────────────────────────────┤
│  nvcr-pytorch:25.09-py3-claude      │  ← Claude Code + keyring
│         -authenticated              │     + OAuth credentials
├─────────────────────────────────────┤
│  nvcr.io/nvidia/pytorch:25.09-py3   │  ← NVIDIA PyTorch base
└─────────────────────────────────────┘
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `docker_build_run_with_claude.sh` | Build base image and run for OAuth authentication |
| `docker_load_and_run_base.sh` | Save/load the authenticated base image |
| `docker_build_run_project.sh` | Build and run project-specific image |
| `docker_run_project.sh` | Run additional containers from existing project image |

## Quick Start

### One-Time Setup: Create Authenticated Base Image

```bash
cd env_scripts

# 1. Build and run base image
./docker_build_run_with_claude.sh

# 2. Inside the container, authenticate Claude Code
claude
# Follow the OAuth flow: open URL in browser, paste code back

# 3. In ANOTHER terminal, save the authenticated image
./docker_load_and_run_base.sh save
```

This creates `~/scratch/docker-images/nvcr-pytorch-25.09-py3-claude-authenticated.tar.gz` (~15-20 GB).

### For Any Project

```bash
# From the project repo (with env_scripts/ in it), run:
./docker_build_run_project.sh

# Claude Code is already authenticated!
claude
```

## Detailed Workflow

### Step 1: Build Base Image with Claude Code

```bash
./docker_build_run_with_claude.sh
```

This builds `nvcr-pytorch:25.09-py3-claude` which includes:
- NVIDIA PyTorch base image
- System tools (nano, gdb, etc.)
- D-Bus and gnome-keyring for credential storage
- Node.js 22.x
- Claude Code CLI

### Step 2: Authenticate Claude Code

Inside the container:
```bash
claude
```

Follow the OAuth flow:
1. Claude will display a URL
2. Open it in your browser
3. Authorize the application
4. Copy the code and paste it back in the terminal

### Step 3: Save Authenticated Image

In a **separate terminal** (while the container is still running):
```bash
./docker_load_and_run_base.sh save
```

This commits the container (with OAuth credentials in `~/.claude/.credentials.json`) and saves it to a tar file.

### Step 4: Use in Projects

From your project repo (containing `env_scripts/`):
```bash
./docker_build_run_project.sh
```

This script:
1. Loads the authenticated base image (if not already in Docker)
2. Builds a project image with dependencies from the **repo root**: if `pyproject.toml` and `uv.lock` exist, installs uv in the image and runs `uv sync --frozen` at container start; else uses `requirements.txt` or `pip install .` from `pyproject.toml`
3. Runs the container with the project directory mounted (same path as on host)
4. Claude Code is already authenticated!

## Adding a New Project

To use this setup for a different project:

1. Copy or adapt `docker_build_run_project.sh` and `docker_run_project.sh` into that project’s `env_scripts/` (or equivalent).
2. Set `PROJECT_IMAGE="your-project:latest"` in both scripts.
3. Ensure the repo has a single source of dependencies at the root:
   - **uv:** `pyproject.toml` and `uv.lock` → image gets uv, container runs `uv sync --frozen` at start.
   - **pip:** `requirements.txt` → image runs `pip install -r requirements.txt`; or only `pyproject.toml` → image runs `pip install .`. In both cases the container runs `pip install -e .` at start from the mount.

No need to hardcode package lists in the script; dependencies are read from the repo.

## On Cluster Nodes

The authenticated base image is saved to `~/scratch/docker-images/`, which is typically shared storage on clusters. When you get a new node:

```bash
# The project script automatically loads the base image from tar if needed
./docker_build_run_project.sh
```

No re-authentication required!

## Running Multiple Containers

To open additional terminal sessions with the same project:

```bash
# First terminal - builds and runs
./docker_build_run_project.sh

# Additional terminals - just runs (no rebuild)
./docker_run_project.sh
./docker_run_project.sh
# ... as many as you need
```

Each container has:
- Full GPU access
- Same project mounts
- Claude Code authentication (from the shared base image)

## Script Details

### docker_build_run_with_claude.sh

- Builds: `nvcr-pytorch:25.09-py3-claude`
- Installs: System tools, D-Bus, gnome-keyring, Node.js, Claude Code CLI
- Creates user matching host UID/GID
- Runs container for authentication (no project mounts)

### docker_load_and_run_base.sh

Commands:
- `save` - Commit running container and save to tar
- `load` - Load authenticated image from tar
- `run` - Run base container (for re-authentication if needed)
- `help` - Show help

Files:
- Image: `nvcr-pytorch:25.09-py3-claude-authenticated`
- Tar: `~/scratch/docker-images/nvcr-pytorch-25.09-py3-claude-authenticated.tar.gz`

### docker_build_run_project.sh

- Uses: `nvcr-pytorch:25.09-py3-claude-authenticated` as base
- Builds: project image (e.g. `my_envtools:latest`); image name is set by `PROJECT_IMAGE` in the script
- Dependencies: read from repo root. If `pyproject.toml` + `uv.lock` → install uv in image, CMD runs `uv sync --frozen` from mounted repo; else `requirements.txt` or `pip install .` from `pyproject.toml`
- Mounts: project directory (repo containing env_scripts) at the same path as on the host
- Optional: `WITH_OLLAMA=true` for local LLM support

## Troubleshooting

### "Authenticated base image not found"

```bash
# Create the authenticated base image first
./docker_build_run_with_claude.sh
# Authenticate claude inside
./docker_load_and_run_base.sh save
```

### Claude Code hangs when starting

The keyring daemon may not have started. Try:
```bash
# Inside the container
eval $(dbus-launch --sh-syntax)
echo "" | gnome-keyring-daemon --unlock --components=secrets
export $(gnome-keyring-daemon --start --components=secrets)
claude
```

### Need to re-authenticate

If your OAuth tokens expire:
```bash
./docker_load_and_run_base.sh run
# Inside: run 'claude' and re-authenticate
# In another terminal: ./docker_load_and_run_base.sh save
```

### Rebuild project image

```bash
docker rmi <your-project-image>:latest
./docker_build_run_project.sh
```
Replace `<your-project-image>` with the value of `PROJECT_IMAGE` in your script (e.g. `my_envtools`).

### Rebuild base image

```bash
docker rmi nvcr-pytorch:25.09-py3-claude
docker rmi nvcr-pytorch:25.09-py3-claude-authenticated
./docker_build_run_with_claude.sh
# Re-authenticate and save
```

## File Locations

| File | Location |
|------|----------|
| OAuth credentials | `~/.claude/.credentials.json` (inside container) |
| Claude config | `~/.claude/` (inside container) |
| Base image tar | `~/scratch/docker-images/nvcr-pytorch-25.09-py3-claude-authenticated.tar.gz` |
| Dockerfiles | `env_scripts/dev.dockerfile.base`, `env_scripts/dev.dockerfile.project` |

## Comparison: Original vs Layered Approach

| Aspect | Original | Layered |
|--------|----------|---------|
| Auth per project | Yes | No (shared base) |
| Saved images | One per project | One base + small project images |
| Storage | ~15GB × N projects | ~15GB base + ~1GB per project |
| Re-auth on new node | Per project | Once for all projects |
| Setup complexity | Lower | Slightly higher initial setup |
