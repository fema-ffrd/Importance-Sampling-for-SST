# Stage 1: Build the obfuscated package
FROM mambaorg/micromamba:1.4.2 AS builder
COPY --chown=$MAMBA_USER:$MAMBA_USER ./.devcontainer/env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# set working directory
WORKDIR /home/mambauser

# get build tools
USER root
RUN apt-get update && apt-get install -y build-essential binutils
USER $MAMBA_USER

# Copy the source code
COPY src /home/mambauser/src

# Install pyarmor and pyinstaller
# RUN micromamba run -n base micromamba install -c conda-forge nuitka
RUN micromamba run -n base python -m pip install --upgrade pip pyarmor pyinstaller

# Compile the package for distribution with pyinstaller
RUN micromamba run -n base pyinstaller -F --collect-data=geopandas src/important-sampling_cli.py

# Generate the obfuscated package with pyarmor
RUN micromamba run -n base pyarmor gen -O obfdist --pack dist/important-sampling_cli src/important-sampling_cli.py

# Stage 2: create the final image
FROM python:3.11-slim AS final

# Set working directory
WORKDIR /dist

# Copy the executable from the builder stage
COPY --from=builder /home/mambauser/dist/important-sampling_cli /dist/important-sampling_cli

# Image metadata
LABEL description="Tools for developing 2D hydraulic models."
LABEL displayName="2D Developer Tools"