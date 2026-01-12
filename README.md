# Raytracer

<div align="center">
  <img width="800" alt="finalScene"
       src="https://github.com/user-attachments/assets/f584bd35-e854-4e84-9ce7-b71fc9419ee7" />
  <br />
  <em>Figure 1: CPU Raytracer render showcasing motion blur, BVH (AABB), texture mapping, Perlin noise, quad primitives, area lights, instancing (translation/rotation), and constant-density media.</em>
</div>

<br />

<div align="center">
  <img width="1200" height="675" alt="cudaSpheres" src="https://github.com/user-attachments/assets/1899bd32-08c5-43dc-830f-e8987ac683ee" />


  <br />
  <em>Figure 2: GPU render showcasing ray–sphere intersection, gamma correction, Lambertian, metallic (with fuzz), and dielectric materials (with total internal reflection and Schlick approximation), dynamic camera motion, and defocus blur. </em>
</div>

## Performance Improvements

- Configuration: `1200×675, 500 spp, max depth 50` <br />
- Base CPU Raytracer with none of the below optimization: `0h 42m 49s 688ms (2569.69 seconds)`

| Acceleration | Speedup | Time |
|--------------|---------|---------------------------------|
| CUDA (no BVH) | 10.44× |  04m 06s 148ms (246.148 seconds) |
| CPU + BVH | 5.67× | 0h 07m 32s 982ms (452.983 seconds) |

*CPU is single-threaded; BVH traversal is used on CPU. CUDA version is brute-force but massively parallel. The timing shows that GPU parallelism can outweigh algorithmic pruning for small-to-medium scenes.*

## Features
This project explores building a Raytracer from scratch, with both CPU and CUDA-accelerated versions.
- The [cuda](https://github.com/pratik-dhende/Raytracer/tree/cuda) branch includes a CUDA-accelerated version.
  - Features
    - Ray Sphere Intersection
    - Gamma Correction
    - Materials supported:
      - Lambertian Material
      - Metal (with fuzziness)
      - Dielectric material with Total Internal Reflection and Schlick Approximation
    - Dynamic Camera
    - Defocus Blur  
- The [main](https://github.com/pratik-dhende/Raytracer/tree/main) branch features a CPU-based raytracer with the same features as CUDA and additionally some new features.
  - Additional features
    - Motion Blur
    - Bounding Volume Heirarchy
      - Axis Aligned Bounding Box 
    - Texture mapping
    - Perline Noise
    - Quadrilateral primitive
      - Ray plane intersection
    - Area Lights
    - Instance translation and rotation
    - Constant Density Media

## Technological Stack
`C++ • CUDA • CMake`

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


    
