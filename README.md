# Raytracer

This project explores building a Raytracer from scratch, with both CPU and CUDA-accelerated versions.
- The [main](https://github.com/pratik-dhende/Raytracer/tree/main) branch features a CPU-based raytracer.
- The [cuda](https://github.com/pratik-dhende/Raytracer/tree/cuda) branch includes a CUDA-accelerated version.

## Technological Stack
`C++ • CUDA • CMake`

## Features
- Ray Sphere Intersection
- Gamma Correction
- Materials supported:
  - Lambertian Material
  - Metal (with fuzziness)
  - Dielectric material with Total Internal Reflection and Schlick Approximation
- Dynamic Camera
- Defocus Blur
- Global illumination - Recursive Ray Tracing with Bounce Limit

## CUDA vs CPU
- The images below were rendered at a resolution of 1200 × 675, using 500 samples per pixel and a maximum ray bounce depth of 50.
- This configuration results in up to 20.25 billion rays traced in the worst case.
- The scene shown in Fig. 1.1 was rendered in under 4 minutes and 32 seconds (Fig. 1.2) with CUDA acceleration.
- For comparison, the scene in Fig. 2.1, rendered with the same configuration as Fig. 1.1, took 32 minutes and 53 seconds on the CPU (Fig. 2.2), achieving a `7.25×` speedup with GPU acceleration.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8ffade08-0ae4-4dc1-95fb-58d2bc7962e2" width="1200"/></td>
    <td><img src="https://github.com/user-attachments/assets/ea087ffc-0d40-4de1-b4a0-a3b1dc3d3f89" width="1200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Figure 1.1: GPU Render Output</em></td>
    <td align="center"><em>Figure 2.1: CPU Render Output</em></td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/1990977c-e7fb-4d90-9b4a-f7871165efe3" width ="1200"/></td>
    <td><img src="https://github.com/user-attachments/assets/3f4d0bf3-ae4b-4092-a8eb-07a819ce2419" width="1200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Figure 1.2: GPU Rendering Time (hh:mm:ss)</em></td>
    <td align="center"><em>Figure 2.2: CPU Rendering Time (hh:mm:ss)</em></td>
  </tr>
</table>
  


## How to Run 
The project uses [CMake](https://cmake.org/) as the meta build system.

### CUDA Accelerated Raytracer
The project requires CUDA 12.6 as the minimum required version.

Checkout `cuda` branch:
```
git checkout cuda
```

Configure CMake by running:
```
cmake -S . -B build
```
For Release build (faster) run:
```
cmake --build build --config Release
```
For Debug build run:
```
cmake --build build --config Debug
```

### CPU Raytracer

Checkout `main` branch:
```
git checkout main
```

Configure CMake by running:
```
cmake -S . -B build
```
For Release build (faster) run:
```
cmake --build build --config Release
```
For Debug build run:
```
cmake --build build --config Debug
```
