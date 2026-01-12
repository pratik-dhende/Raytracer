# Raytracer

<div align="center">
  <img width="800" alt="finalScene"
       src="https://github.com/user-attachments/assets/f584bd35-e854-4e84-9ce7-b71fc9419ee7" />
  <br />
  <em>Figure 1: CPU Raytracer render showcasing motion blur, BVH (AABB), texture mapping, Perlin noise, quad primitives, area lights, instancing (translation/rotation), and constant-density media.</em>
</div>

## Performance Improvements

| Acceleration | Speedup |
|--------------|---------|
| CUDA        | 7.25×   |
| BVH          | 4.7×    |

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

## BVH vs Non-BVH
- The images below were rendered at a resolution of 400 × 225, using 100 samples per pixel and a maximum ray bounce depth of 50.
- This configuration results in up to 450 million rays traced in the worst case.
- The scene shown in Fig. 2.1 was rendered in under 14.5 seconds (Fig. 2.2) with BVH acceleration.
- For comparison, the scene in Fig. 3.1, rendered with the same configuration as Fig. 2.1, took 1 minute and 8 seconds with the BVH (Fig. 3.2), achieving a `4.7x` speedup with BVH acceleration.

<table>
  <tr>
    <td><img width="400" height="225" alt="bvh" src="https://github.com/user-attachments/assets/c186b57d-8f1f-4cd9-bf6e-95557ad5c094" /></td>
    <td><img width="400" height="225" alt="no_bvh" src="https://github.com/user-attachments/assets/4030ebdb-a5e5-4716-ae07-d926a39ae4e1" /></td>
</td>
  </tr>
  <tr>
    <td align="center"><em>Figure 2.1: BVH Render Output</em></td>
    <td align="center"><em>Figure 3.1: Non-BVH Render Output</em></td>
  </tr>

  <tr>
    <td><img width="352" height="65" alt="bvh" src="https://github.com/user-attachments/assets/d67e48d2-d04a-4a37-8d02-014b39815292" />

</td>
    <td><img width="363" height="58" alt="no_bvh" src="https://github.com/user-attachments/assets/8cb23bf0-e8e0-4aef-a163-6fd7deb7ef9b" />
</td>
  </tr>
  <tr>
    <td align="center"><em>Figure 2.2: BVH Rendering Time (hh:mm:ss)</em></td>
    <td align="center"><em>Figure 3.2: Non-BVH Rendering Time (hh:mm:ss)</em></td>
  </tr>
</table>

## CUDA vs CPU
- The images below were rendered at a resolution of 1200 × 675, using 500 samples per pixel and a maximum ray bounce depth of 50.
- This configuration results in up to 20.25 billion rays traced in the worst case.
- The scene shown in Fig. 4.1 was rendered in under 4 minutes and 32 seconds (Fig. 4.2) with CUDA acceleration.
- For comparison, the scene in Fig. 5.1, rendered with the same configuration as Fig. 5.1, took 32 minutes and 53 seconds on the CPU (Fig. 5.2), achieving a `7.25×` speedup with GPU acceleration.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8ffade08-0ae4-4dc1-95fb-58d2bc7962e2" width="1200"/></td>
    <td><img src="https://github.com/user-attachments/assets/ea087ffc-0d40-4de1-b4a0-a3b1dc3d3f89" width="1200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Figure 4.1: GPU Render Output</em></td>
    <td align="center"><em>Figure 5.1: CPU Render Output</em></td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/1990977c-e7fb-4d90-9b4a-f7871165efe3" width ="1200"/></td>
    <td><img src="https://github.com/user-attachments/assets/3f4d0bf3-ae4b-4092-a8eb-07a819ce2419" width="1200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Figure 4.2: GPU Rendering Time (hh:mm:ss)</em></td>
    <td align="center"><em>Figure 5.2: CPU Rendering Time (hh:mm:ss)</em></td>
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


    
