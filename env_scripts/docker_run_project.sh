#!/bin/bash
# Usage: ./docker_run_project.sh [WITH_OLLAMA=true]
#
# Runs a container from the already-built project image.
# Use this to open additional terminals/containers for the same project.
#
# Prerequisites:
#   The project image must already exist. Build it first with:
#   ./docker_build_run_project.sh
#
# If WITH_OLLAMA=true, connects to the existing Ollama network (if running).

set -e

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$DIR"

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)

# Project-specific image name (must match docker_build_run_project.sh)
PROJECT_IMAGE="ksearch-env:latest"

# Check if project image exists
if [ -z "$(docker images -q ${PROJECT_IMAGE} 2>/dev/null)" ]; then
    echo "ERROR: Project image '${PROJECT_IMAGE}' not found!"
    echo ""
    echo "Build the project image first by running:"
    echo "  ./docker_build_run_project.sh"
    exit 1
fi

# Mount configurations
CODE_FOLDER=/home/${MY_UNAME}/code
MOUNT_CODE_FOLDER=""
if [ -d "${CODE_FOLDER}" ]; then
    MOUNT_CODE_FOLDER="--mount type=bind,source=${CODE_FOLDER},target=${CODE_FOLDER}"
fi

# Claude Code authentication: pass ANTHROPIC_API_KEY if set on host (optional)
CLAUDE_ENV_VARS=""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    CLAUDE_ENV_VARS="-e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
fi

# Parse WITH_OLLAMA argument (default: false)
WITH_OLLAMA=false
if [[ "$1" == "WITH_OLLAMA="* ]]; then
    WITH_OLLAMA="${1#WITH_OLLAMA=}"
    shift
fi

# Network argument - connect to existing llmnet if WITH_OLLAMA is true
NETWORK_ARG=""
if [[ "$WITH_OLLAMA" == "true" || "$WITH_OLLAMA" == "True" ]]; then
    if docker network inspect llmnet >/dev/null 2>&1; then
        NETWORK_ARG="--network llmnet"
        echo "Connecting to existing llmnet network..."
    else
        echo "Warning: llmnet network not found. Run docker_build_run_project.sh WITH_OLLAMA=true first."
    fi
fi

echo "Starting container (image CMD: uv sync or pip install -e . then bash)..."
echo ""

docker run \
    --gpus "device=all" \
    $NETWORK_ARG \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --mount type=bind,source=${DIR}/..,target=${DIR}/.. \
    ${CLAUDE_ENV_VARS} \
    --shm-size=8g \
    ${PROJECT_IMAGE}
 