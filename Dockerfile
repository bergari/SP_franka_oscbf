# Use a stable base image with CUDA 12.1. This is compatible with recent ZED SDKs (like 5.2)
# and has official, stable PyTorch wheels. This image is based on Ubuntu 22.04 (for ROS2 Humble).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set up environment and install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
# Note: tzdata is added to prevent interactive prompts during installation.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    python3-pip \
    python3-venv \
    lsb-release \
    curl \
    gnupg2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-humble-ros-base

# Clone 4D Humans from Github
RUN git clone https://github.com/shubham-goel/4D-Humans.git /opt/4D-Humans
WORKDIR /opt/4D-Humans

# Create the data directory and inject the SMPL model from host
RUN mkdir -p data
COPY basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/

# Install PyTorch and 4D-Humans dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --break-system-packages
RUN pip install .[all] --break-system-packages

# Download the remaining 4D-Humans weights
RUN python3 -c "from hmr2.models import download_models, check_smpl_exists; download_models(); check_smpl_exists()"

# Setup custom Ros2 workspace
WORKDIR /ros2_ws

# Copy Franka/YOLO package into the container's source folder
COPY . src/SP_franka_oscbf/

# Build the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"
# Source both the core ROS installation and custom workspace
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec \"$@\""]

# Launch specific node by default
CMD ["ros2", "run", "human_tracker", "skeleton_tracker_node_4DHumans"]