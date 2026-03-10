#!/bin/bash
# Usage: ./docker_build_run_with_claude.sh
#
# Builds a base image with PyTorch + Claude Code CLI + keyring support.
# This image is meant to be authenticated once and saved for reuse across projects.
#
# Workflow:
#   1. Run this script to build and launch the base container
#   2. Inside the container, run 'claude' and authenticate via OAuth
#   3. In another terminal, run './docker_load_and_run_base.sh save' to save the authenticated image
#   4. For projects, use docker_build_run_project.sh which builds on top of the authenticated base

set -e

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$DIR"

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)

# Base image configuration
BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
IMAGE_NAME="nvcr-pytorch:25.09-py3-claude"

echo "Building base image: ${IMAGE_NAME}"
echo "This image includes PyTorch + Claude Code CLI + keyring support"

if [ -z "$(docker images -q ${IMAGE_NAME})" ]; then
    FILE=dev.dockerfile.base

    echo "FROM $BASE_IMAGE" > $FILE

    # System packages
    echo "  RUN apt-get update" >> $FILE
    echo "  RUN apt-get -y install nano gdb time" >> $FILE
    echo "  RUN apt-get -y install nvidia-cuda-gdb" >> $FILE
    echo "  RUN apt-get -y install sudo" >> $FILE

    # D-Bus and gnome-keyring for Claude Code OAuth token storage
    echo "  RUN apt-get -y install dbus dbus-x11 gnome-keyring libsecret-1-0 libsecret-tools" >> $FILE

    # Node.js (required for Claude Code CLI)
    echo "  RUN apt-get -y install curl" >> $FILE
    echo "  RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -" >> $FILE
    echo "  RUN apt-get -y install nodejs" >> $FILE

    # Install Claude Code CLI globally (as root, before switching to user)
    echo "  RUN npm install -g @anthropic-ai/claude-code" >> $FILE

    # Create user with same UID/GID as host user
    echo "  RUN (groupadd -g $MY_GID $MY_UNAME || true) && useradd --uid $MY_UID -g $MY_GID --no-log-init --create-home $MY_UNAME && (echo \"${MY_UNAME}:password\" | chpasswd) && (echo \"${MY_UNAME} ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers)" >> $FILE

    # Increase inotify watches for better file watching
    echo "  RUN echo \"fs.inotify.max_user_watches=524288\" >> /etc/sysctl.conf" >> $FILE
    echo "  RUN sysctl -p || true" >> $FILE

    # Switch to user
    echo "  USER $MY_UNAME" >> $FILE
    echo "  COPY docker.bashrc.base /home/${MY_UNAME}/.bashrc" >> $FILE

    echo "  WORKDIR /home/${MY_UNAME}" >> $FILE

    # CMD: Start D-Bus and keyring, then bash
    cat >> $FILE << 'DOCKERFILE_CMD'
  CMD ["/bin/bash", "-c", "eval $(dbus-launch --sh-syntax) && echo '' | gnome-keyring-daemon --unlock --components=secrets 2>/dev/null; export $(gnome-keyring-daemon --start --components=secrets 2>/dev/null); exec /bin/bash"]
DOCKERFILE_CMD

    docker buildx build -f $FILE -t ${IMAGE_NAME} .
    echo ""
    echo "Base image built successfully: ${IMAGE_NAME}"
else
    echo "Base image already exists: ${IMAGE_NAME}"
fi

echo ""
echo "Starting container for Claude Code authentication..."
echo "Once inside, run 'claude' to authenticate via OAuth."
echo "After authentication, open another terminal and run:"
echo "  ./docker_load_and_run_base.sh save"
echo ""

# Run container without any project mounts - just for authentication
docker run \
    --gpus "device=all" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --shm-size=8g \
    ${IMAGE_NAME}
