# Use your existing heavy image as the foundation
FROM sp_franka_oscbf-vision_pipeline:latest

# Switch to root to install the missing software renderer
USER root
RUN apt-get update && apt-get install -y libosmesa6 libosmesa6-dev

# Upgrade the Python wrapper so it recognizes the modern OSMesa functions
RUN pip3 install --upgrade PyOpenGL PyOpenGL_accelerate