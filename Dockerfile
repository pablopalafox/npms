FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# FROM nvidia/cuda:11.0-devel-ubuntu20.04
# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# FROM nvcr.io/nvidia/pytorch:20.01-py3

ENV DEBIAN_FRONTEND noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    xvfb \
    curl \
    wget \
    llvm \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    mono-mcs \
    python3-venv \
    python3-dev \
    vim \
    libglu1-mesa-dev \
    libsdl2-dev \ 
    libc++-7-dev \ 
    libc++abi-7-dev \ 
    ninja-build \ 
    libxi-dev \ 
    libtbb-dev \
    libosmesa6-dev \
    libusb-1.0-0-dev \
    manpages-dev \
    --reinstall build-essential \
    llvm-6.0 \
    llvm-6.0-tools \
    freeglut3 \
    freeglut3-dev \
    ffmpeg -y \
    && rm -rf /var/lib/apt/lists/*

# Install OSMesa for pyrender
RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb \
    && dpkg -i ./mesa_18.3.3-0.deb || true \
    && apt-get install -f \
    && rm ./mesa_18.3.3-0.deb

# Set python3 as main python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Activate virtuaenv (https://pythonspeed.com/articles/activate-virtualenv-dockerfile/)
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN python -m pip install --upgrade pip

# Create a working directory
WORKDIR /app

# Install requirements
COPY docker_requirements.txt . 
RUN pip install -r docker_requirements.txt
RUN rm requirements.txt

############################################
# Install dependencies in 'external'
############################################
# - First, copy the 'external' folder directly to the container
ADD external ./external

# - Clone Eigen
WORKDIR /app/external
RUN git clone https://gitlab.com/libeigen/eigen.git

# - PyMarchingCubes (a fork of PyMCubes, by Justus Thies - 
#   we have a local copy here because we changed some path in a header file)
WORKDIR /app/external/PyMarchingCubes
RUN python setup.py install

# - BUild gaps (Thomas Funkhouser)
WORKDIR /app/external
RUN chmod +x build_gaps.sh && ./build_gaps.sh
RUN chmod +x gaps_is_installed.sh && ./gaps_is_installed.sh

# - Some libs from IFNet (Julian Chibane)
WORKDIR /app/external/libmesh/
RUN python setup.py build_ext --inplace
WORKDIR /app/external/libvoxelize/
RUN python setup.py build_ext --inplace

# - Install some cpp helper functions
WORKDIR /app/external/csrc
RUN python setup.py install

# Go back to the main directory
WORKDIR /app
