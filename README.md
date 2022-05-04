# project

My final project for EN605.617 - intro to GPU programming

# Source structure

## `src`

This directory contains the main driver for the project, as well as any other top-level relevant source files

All subdirectories are considered to be "internal" libraries, and help to provide source orgainization, and are not intended to be extracted as standalone libraries.

## `src/fftw`

This directory contains code related to the `FFTW`-based cpu benchmark

## `src/cuda`

This directory contains code related to the `cuFFT`-based gpu benchmark

## `src/io`

This directory contains any io-related code

# Building and running

This project was built and run exclusively in [gpu-aware docker containers](https://github.com/NVIDIA/nvidia-docker), primarily using [vscode development containers](https://code.visualstudio.com/docs/remote/containers). This workflow is highly recommended, but not required.

## Manually building

This project is targeted for cuda version 11.x, and uses `cmake` to build. The minimum required version of `cmake` is 3.8, which virtually any modern os has available.

To configure the `cmake` project, run:

```
cmake . -Bbuild -DCMAKE_BUILD_TYPE:STRING=Release
```

at the repo root.

To build, run

```
cmake --build build --target en605.617-project -j $(nproc)
```

To execute, run

```
cd build/src
./en605.617-project
```
