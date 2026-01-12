# Raytracer

<div align="center">
  <img width="800" alt="finalScene"
       src="https://github.com/user-attachments/assets/f584bd35-e854-4e84-9ce7-b71fc9419ee7" />
  <br />
  <em>Figure 1: CPU render showcasing motion blur, BVH (AABB), texture mapping, Perlin noise, quad primitives, area lights, instancing, and constant-density media.</em>
</div>

<br />

<div align="center">
  <img width="1200" height="675" alt="cudaSpheres" src="https://github.com/user-attachments/assets/1899bd32-08c5-43dc-830f-e8987ac683ee" />


  <br />
  <em>Figure 2: GPU render showcasing ray–sphere intersection, gamma correction, Lambertian, metallic (with fuzz), and dielectric materials (with total internal reflection and Schlick approximation), dynamic camera, and defocus blur.</em>
</div>

## Performance Comparison

| Configuration       | Value      |
|--------------------|-----------|
| Resolution          | 1200×675  |
| Samples Per Pixel   | 500       |
| Max Bounce Depth    | 50        |

| Acceleration | Speedup | Time (hh:mm:ss) |
|--------------|---------|---------------------------------|
| `None` | `1x` | `00::42::49::688 (2569.69 seconds)` |
| `CPU + BVH` | `5.67×` | `00:07:32.982 (452.983 seconds)` |
| `CUDA (no BVH)` | `10.44×` | `00:04:06.148 (246.148 seconds)` |

## Features

This project explores building a Raytracer from scratch, with both CPU and CUDA-accelerated versions.

| Feature | CPU Branch | CUDA Branch |
|---------|------------|------------|
| Ray–Sphere Intersection | ✅ | ✅ |
| Gamma Correction | ✅ | ✅ |
| Lambertian Material | ✅ | ✅ |
| Metal (with fuzz) | ✅ | ✅ |
| Dielectric (TIR + Schlick) | ✅ | ✅ |
| Dynamic Camera | ✅ | ✅ |
| Defocus Blur | ✅ | ✅ |
| Motion Blur | ✅ | ❌ |
| Bounding Volume Hierarchy (AABB) | ✅ | ❌ |
| Texture Mapping | ✅ | ❌ |
| Perlin Noise | ✅ | ❌ |
| Quadrilateral Primitive (Ray–Plane Intersection) | ✅ | ❌ |
| Area Lights | ✅ | ❌ |
| Instance Translation/Rotation | ✅ | ❌ |
| Constant-Density Media | ✅ | ❌ |


## Technological Stack
`C++ • CUDA • CMake`

## How to Run 
- The project uses [CMake](https://cmake.org/) as the meta build system.
- CUDA version >= 12.6 and MSVC are required for the [cuda](https://github.com/pratik-dhende/Raytracer/tree/cuda?tab=readme-ov-file) branch.

### Check out the respective branch:
```bash
git checkout main    # If CPU
git checkout cuda    # If CUDA
```

### Configure CMake
```
cmake -S . -B build
```
### Release
```
cmake --build build --config Release
```
### Debug
```
cmake --build build --config Debug
```



    
