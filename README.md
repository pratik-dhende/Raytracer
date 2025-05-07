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
- The image below was rendered at a resolution of 1200 × 675 with 500 samples per pixel and a maximum ray bounce limit of 50.
- This results in a total of approximately 20.25 billion rays traced.
- The scene was rendered in under 4 minutes and 32 seconds using CUDA. (Timing screenshot provided below the image.)
  
![final-scene-release-render](https://github.com/user-attachments/assets/8ffade08-0ae4-4dc1-95fb-58d2bc7962e2)
  
![final-scene-release-time-highlighted](https://github.com/user-attachments/assets/831a5f77-3f84-495b-9e12-73d7f61703cb)


## How to Run
The project uses [CMake](https://cmake.org/) as the meta build system.

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


    
