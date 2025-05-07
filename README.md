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

## Render Output
- The images below are rendered at a resolution of 1200 × 675 with 500 samples per pixel and a maximum ray bounce limit of 50.
- This results in total of 20.25 billion rays traced in the worst case.
- The scene was rendered in under 4 minutes and 32 seconds (*Fig. 1.2*) using CUDA acceleration.
- For comparison, the same image takes 32 minutes 53 seconds (*Fig. 2.2*) on CPU.
- Speedup achieved is 7.25x

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
    <td align="center"><em>Figure 1.2: GPU Rendering Time</em></td>
    <td align="center"><em>Figure 2.2: CPU Rendering Time</em></td>
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


    
