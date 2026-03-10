#!/bin/bash
# Usage: ./docker_build_run_project.sh [WITH_OLLAMA=true]
#
# Builds a project-specific image on top of the authenticated Claude base image.
# Dependencies are taken from the repo root: if pyproject.toml + uv.lock exist, uses uv
# (install uv in image, run "uv sync --frozen" at container start); else requirements.txt
# or pip install from pyproject.toml.
#
# Prerequisites:
#   1. The authenticated base image must exist. Create it by running:
#      ./docker_build_run_with_claude.sh
#      # Authenticate Claude inside the container
#      ./docker_load_and_run_base.sh save
#
# If WITH_OLLAMA=true, the script will setup Ollama and use the llmnet network.

set -e

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT="${DIR}/.."
cd "$DIR"

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)

# Base image configuration (authenticated Claude base)
BASE_IMAGE_NAME="nvcr-pytorch:25.09-py3-claude-authenticated"
IMAGE_TAR_DIR="${HOME}/scratch/docker-images"
IMAGE_TAR_PATH="${IMAGE_TAR_DIR}/nvcr-pytorch-25.09-py3-claude-authenticated.tar.gz"

# Project-specific image name
PROJECT_IMAGE="ksearch-env:latest"

# Ensure .cursor-server directory exists
mkdir -p "${DIR}/.cursor-server"
LINK=$(realpath --relative-to="/home/${MY_UNAME}" "$DIR" -s 2>/dev/null || true)

# Function to ensure base image is available
ensure_base_image() {
    # Check if authenticated base image exists in docker
    if [ -n "$(docker images -q ${BASE_IMAGE_NAME} 2>/dev/null)" ]; then
        echo "Authenticated base image found: ${BASE_IMAGE_NAME}"
        return 0
    fi

    # Try to load from tar file
    if [ -f "${IMAGE_TAR_PATH}" ]; then
        echo "Loading authenticated base image from ${IMAGE_TAR_PATH}..."
        docker load < "${IMAGE_TAR_PATH}"
        echo "Base image loaded successfully"
        return 0
    fi

    echo "ERROR: Authenticated base image not found!"
    echo ""
    echo "The authenticated Claude base image is required."
    echo "To create it, run:"
    echo "  1. ./docker_build_run_with_claude.sh"
    echo "  2. Inside container: run 'claude' and authenticate via OAuth"
    echo "  3. In another terminal: ./docker_load_and_run_base.sh save"
    echo ""
    exit 1
}

# Ensure base image is available
ensure_base_image

# Build project-specific image if it doesn't exist
if [ -z "$(docker images -q ${PROJECT_IMAGE} 2>/dev/null)" ]; then
    echo "Building project image: ${PROJECT_IMAGE}"
    FILE="${DIR}/dev.dockerfile.project"

    USE_UV=false
    [ -f "${REPO_ROOT}/pyproject.toml" ] && [ -f "${REPO_ROOT}/uv.lock" ] && USE_UV=true

    {
        echo "FROM ${BASE_IMAGE_NAME}"
        echo "  USER root"
        echo "  RUN mkdir -p ${DIR}"
        if [ -n "$LINK" ]; then
            echo "  RUN ln -sf ${LINK}/.cursor-server /home/${MY_UNAME}/.cursor-server 2>/dev/null || true"
        fi
        echo "  USER ${MY_UNAME}"
    } > "$FILE"

    if [ "$USE_UV" = true ]; then
        echo "Using uv: install uv in image; CMD will run 'uv sync --frozen' from mounted repo."
        {
            echo "  RUN pip install --no-cache-dir uv"
            echo "  ENV UV_LINK_MODE=copy"
            echo "  WORKDIR ${DIR}/.."
            echo '  CMD ["/bin/bash", "-c", "eval $(dbus-launch --sh-syntax) && echo \"\" | gnome-keyring-daemon --unlock --components=secrets 2>/dev/null; export $(gnome-keyring-daemon --start --components=secrets 2>/dev/null); echo \"Running uv sync --frozen...\"; uv sync --frozen 2>&1 || true; source .venv/bin/activate 2>/dev/null || true; exec /bin/bash"]'
        } >> "$FILE"
    else
        if [ -f "${REPO_ROOT}/requirements.txt" ]; then
            echo "Using requirements.txt from repo root."
            {
                echo "  COPY requirements.txt ./"
                echo "  RUN pip install --no-cache-dir -r requirements.txt"
                echo "  WORKDIR ${DIR}/.."
                echo '  CMD ["/bin/bash", "-c", "eval $(dbus-launch --sh-syntax) && echo \"\" | gnome-keyring-daemon --unlock --components=secrets 2>/dev/null; export $(gnome-keyring-daemon --start --components=secrets 2>/dev/null); echo \"Installing project in editable mode...\"; pip install --no-deps -e . 2>&1 || true; exec /bin/bash"]'
            } >> "$FILE"
        else
            echo "Using pyproject.toml (no uv.lock): copying project and running pip install ."
            {
                echo "  COPY . ./"
                echo "  RUN pip install --no-cache-dir ."
                echo "  WORKDIR ${DIR}/.."
                echo '  CMD ["/bin/bash", "-c", "eval $(dbus-launch --sh-syntax) && echo \"\" | gnome-keyring-daemon --unlock --components=secrets 2>/dev/null; export $(gnome-keyring-daemon --start --components=secrets 2>/dev/null); echo \"Installing project in editable mode...\"; pip install --no-deps -e . 2>&1 || true; exec /bin/bash"]'
            } >> "$FILE"
        fi
    fi

    docker buildx build -f "$FILE" -t "${PROJECT_IMAGE}" "${REPO_ROOT}"
    echo "Project image built successfully: ${PROJECT_IMAGE}"
else
    echo "Project image already exists: ${PROJECT_IMAGE}"
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

# Only run Ollama setup if WITH_OLLAMA is true
if [[ "$WITH_OLLAMA" == "true" || "$WITH_OLLAMA" == "True" ]]; then
    docker network inspect llmnet >/dev/null 2>&1 || docker network create llmnet

    MODEL_NAME=qwen3:8b
    docker run -d --rm --gpus all --name ollama --network llmnet -p 11434:11434 ollama/ollama

    until curl -s http://localhost:11434 | grep -q 'Ollama'; do
        echo "Waiting for Ollama to be ready..."
        sleep 1
    done

    docker exec -it ollama ollama pull ${MODEL_NAME}
fi

# Network argument
NETWORK_ARG=""
if [[ "$WITH_OLLAMA" == "true" || "$WITH_OLLAMA" == "True" ]]; then
    NETWORK_ARG="--network llmnet"
fi

echo ""
echo "Starting ${PROJECT_IMAGE} container..."
echo "Claude Code should already be authenticated from the base image."
echo ""

docker run \
    --gpus "device=all" \
    $NETWORK_ARG \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --mount type=bind,source=${DIR}/..,target=${DIR}/.. \
    ${CLAUDE_ENV_VARS} \
    --shm-size=8g \
    ${PROJECT_IMAGE}


#     ${MOUNT_CODE_FOLDER} \


cd - > /dev/null
