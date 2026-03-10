#!/bin/bash
# Usage: ./docker_load_and_run_base.sh [COMMAND]
#
# Commands:
#   save   - Commit a running base container and save to tar file
#   load   - Load the authenticated base image from tar file
#   run    - Load (if needed) and run the base container for re-authentication
#   help   - Show this help message
#
# This script manages the authenticated base image that can be shared across projects.
#
# Workflow:
#   1. Run ./docker_build_run_with_claude.sh and authenticate Claude
#   2. Run ./docker_load_and_run_base.sh save (in another terminal)
#   3. For any project, the docker_build_run_project.sh will use this authenticated base

set -e

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$DIR"

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)

# Configuration - Base image naming
IMAGE_BASE="nvcr-pytorch"
IMAGE_TAG="25.09-py3-claude"
IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"
IMAGE_NAME_AUTH="${IMAGE_BASE}:${IMAGE_TAG}-authenticated"

# Where to save the authenticated image tar
IMAGE_TAR_DIR="${HOME}/scratch/docker-images"
IMAGE_TAR_PATH="${IMAGE_TAR_DIR}/${IMAGE_BASE}-${IMAGE_TAG}-authenticated.tar.gz"

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Manages the authenticated base image with Claude Code."
    echo ""
    echo "Commands:"
    echo "  save   Commit running container and save authenticated image to tar"
    echo "  load   Load the authenticated image from tar file"
    echo "  run    Load (if needed) and run container for re-authentication"
    echo "  help   Show this help message"
    echo ""
    echo "Configuration:"
    echo "  Base image:          ${IMAGE_NAME}"
    echo "  Authenticated image: ${IMAGE_NAME_AUTH}"
    echo "  Image tar path:      ${IMAGE_TAR_PATH}"
    echo ""
    echo "Workflow:"
    echo "  1. Build and authenticate:"
    echo "     ./docker_build_run_with_claude.sh"
    echo "     # Inside container: run 'claude' and authenticate"
    echo ""
    echo "  2. Save authenticated image (in another terminal):"
    echo "     ./docker_load_and_run_base.sh save"
    echo ""
    echo "  3. Use in projects:"
    echo "     ./docker_build_run_project.sh"
}

save_image() {
    # Find running container from base image
    CONTAINER_ID=$(docker ps --filter "ancestor=${IMAGE_NAME}" --filter "ancestor=${IMAGE_NAME_AUTH}" -q | head -1)

    if [ -z "$CONTAINER_ID" ]; then
        echo "ERROR: No running ${IMAGE_BASE} container found"
        echo ""
        echo "Make sure you have a container running from docker_build_run_with_claude.sh"
        echo "You can check running containers with: docker ps"
        exit 1
    fi

    echo "Found running container: ${CONTAINER_ID}"

    # Create output directory if needed
    mkdir -p "${IMAGE_TAR_DIR}"

    # Commit the container
    echo "Committing container to ${IMAGE_NAME_AUTH}..."
    docker commit "${CONTAINER_ID}" "${IMAGE_NAME_AUTH}"

    # Save to tar.gz
    echo "Saving image to ${IMAGE_TAR_PATH}..."
    echo "This may take several minutes for large images..."
    docker save "${IMAGE_NAME_AUTH}" | gzip > "${IMAGE_TAR_PATH}"

    # Show size
    SIZE=$(du -h "${IMAGE_TAR_PATH}" | cut -f1)
    echo ""
    echo "Done! Authenticated base image saved to: ${IMAGE_TAR_PATH}"
    echo "Image size: ${SIZE}"
    echo ""
    echo "You can now use this image in any project with docker_build_run_project.sh"
}

load_image() {
    # Check if authenticated image already exists
    if [ -n "$(docker images -q ${IMAGE_NAME_AUTH} 2>/dev/null)" ]; then
        echo "Authenticated image ${IMAGE_NAME_AUTH} already loaded in docker"
        return 0
    fi

    # Check if tar file exists
    if [ ! -f "${IMAGE_TAR_PATH}" ]; then
        echo "ERROR: Authenticated image tar not found at ${IMAGE_TAR_PATH}"
        echo ""
        echo "To create the authenticated base image:"
        echo "  1. Run ./docker_build_run_with_claude.sh"
        echo "  2. Inside container, run 'claude' and authenticate"
        echo "  3. In another terminal, run ./docker_load_and_run_base.sh save"
        exit 1
    fi

    echo "Loading authenticated image from ${IMAGE_TAR_PATH}..."
    echo "This may take a few minutes..."
    docker load < "${IMAGE_TAR_PATH}"
    echo "Image loaded successfully: ${IMAGE_NAME_AUTH}"
}

run_container() {
    load_image

    echo ""
    echo "Starting base container for re-authentication or testing..."
    echo "This container has no project mounts."
    echo ""

    docker run \
        --gpus "device=all" \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
        --shm-size=8g \
        ${IMAGE_NAME_AUTH} \
        /bin/bash -c 'eval $(dbus-launch --sh-syntax) && \
            echo "" | gnome-keyring-daemon --unlock --components=secrets 2>/dev/null; \
            export $(gnome-keyring-daemon --start --components=secrets 2>/dev/null); \
            exec /bin/bash'
}

# Main command dispatch
COMMAND="${1:-help}"

case "$COMMAND" in
    save)
        save_image
        ;;
    load)
        load_image
        ;;
    run)
        run_container
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
