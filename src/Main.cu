#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "Cuda.h"

#include <iostream>

namespace Cuda {
    __device__
    Color rayColor(const Ray& ray, const Hittable& world, int depth, curandState& randState) {
        Color pixelColor(1.0);
        Ray incidentRay = ray;

        Color attenuation;
        Ray scatteredRay;
        HitInfo hitInfo;

        while(depth > 0) {
            if(world.hit(incidentRay, Interval(0.001, POSITIVE_INFINITY), hitInfo)) {
                if (hitInfo.material->scatter(incidentRay, hitInfo, attenuation, scatteredRay, randState)) {
                    pixelColor *= attenuation;
                    incidentRay = scatteredRay;
                }
                else {
                    pixelColor = Color(0.0);
                    break;
                }
            }
            else {
                Vec3 unitDirection = incidentRay.direction().normalized();
                auto a = 0.5 * (unitDirection.y() + 1.0);
                pixelColor *= (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0);
                break;
            }
            --depth;
        }

        return depth ? pixelColor : Color(0.0);
    }

    __global__
    void render(Scene** d_world, Vec3* d_framebuffer, Camera* d_camera, curandState *randState) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (j >= d_camera->imageWidth || i >= d_camera->getFrameBufferHeight()) {
            return;
        }
        
        int pixelIndex = i * d_camera->imageWidth + j;

        curandState localRandState = randState[pixelIndex];
        Color pixelColor = Color(0.0);

        for(int sample = 0; sample < d_camera->samplesPerPixel; ++sample) {
            pixelColor += rayColor(d_camera->sampleRay(j, i, localRandState), **d_world, d_camera->maxDepth, localRandState);
        }
        pixelColor *= d_camera->getPixelsPerSample();
        
        randState[pixelIndex] = localRandState;
        
        d_framebuffer[pixelIndex] = pixelColor;
    }

    __global__
    void initPixelCurandState(curandState* d_randState, Camera* d_camera) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (j >= d_camera->imageWidth || i >= d_camera->getFrameBufferHeight()) {
            return;
        }
        
        int pixelIndex = i * d_camera->imageWidth + j;
        curand_init(1984 + pixelIndex, 0, 0, &d_randState[pixelIndex]);
    }

    __global__
    void initSceneCurandState(curandState* d_randState) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }
        curand_init(1984, 0, 0, d_randState);
    }

    __global__ 
    void createScene(Scene** d_world, curandState* d_randState) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        *d_world = new Scene(22 * 22 + 3);
        Scene* world = *d_world;

        curandState localRandState = *d_randState;
    
        auto groundMaterial = new Lambertian(Color(0.5, 0.5, 0.5));
        world->add(new Sphere(Point3(0.0 ,-1000.0 ,0.0), 1000.0, groundMaterial));
    
        for (int a = -11; a < 11; ++a) {
            for (int b = -11; b < 11; ++b) {
                auto chosenMaterial = Cuda::random(localRandState);
                Point3 center(a + 0.9 * Cuda::random(localRandState), 0.2, b + 0.9 * Cuda::random(localRandState));
    
                if ((center - Point3(4.0, 0.2, 0.0)).magnitude() > 0.9) {
                    Material* sphereMaterial;
    
                    if (chosenMaterial < 0.8) {
                        // Diffuse
                        auto albedo = Color::random(localRandState) * Color::random(localRandState);
                        sphereMaterial = new Lambertian(albedo);
                        world->add(new Sphere(center, 0.2, sphereMaterial));
                    } 
                    else if (chosenMaterial < 0.95) {
                        // Metal
                        auto albedo = Color::random(0.5, 1, localRandState);
                        auto fuzz = Cuda::random(0, 0.5, localRandState);
                        sphereMaterial = new Metal(albedo, fuzz);
                        world->add(new Sphere(center, 0.2, sphereMaterial));
                    } 
                    else {
                        // Glass
                        sphereMaterial = new Dielectric(1.5);
                        world->add(new Sphere(center, 0.2, sphereMaterial));
                    }
                }
            }
        }
    
        auto material1 = new Dielectric(1.5);
        world->add(new Sphere(Point3(0.0, 1.0, 0.0), 1.0, material1));
    
        auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        world->add(new Sphere(Point3(-4.0, 1.0, 0.0), 1.0, material2));
    
        auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
        world->add(new Sphere(Point3(4.0, 1.0, 0.0), 1.0, material3));

        *d_randState = localRandState;
    }

    __global__
    void deleteScene(Scene** d_world) {
        if (*d_world) {
            delete *d_world;
        }
    }
}

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    // Create a scene in device memory
    Cuda::SmartPointer<Scene*> d_world(1, false);
    Cuda::SmartPointer<curandState> d_sceneRandState(1, false);

    Cuda::initSceneCurandState<<<1, 1>>>(d_sceneRandState.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Cuda::createScene<<<1, 1>>>(d_world.get(), d_sceneRandState.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Init camera parameters
    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 1200;
    camera.samplesPerPixel = 500;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(13.0, 2.0, 3.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.6;
    camera.focusDistance = 10.0;

    camera.init();

    std::clog << "Width: " << camera.imageWidth << " Height: " << camera.getFrameBufferHeight() << " Total: " << camera.imageWidth * camera.getFrameBufferHeight() << "\n";

    // Allocate and initialize camera on device
    Cuda::SmartPointer<Camera> d_camera(1, false);
    cudaMemcpy(d_camera.get(), &camera, sizeof(Camera), cudaMemcpyHostToDevice);

    // Allocate framebuffer on device
    int framebufferSize = camera.imageWidth * camera.getFrameBufferHeight();
    Cuda::SmartPointer<Vec3> d_frameBuffer(framebufferSize, true);

    // Compute dimensions for render
    int threadsX = 8;
    int threadsY = 8;
    dim3 blockDimensions(threadsX, threadsY);
    dim3 gridDimensions((camera.imageWidth + threadsX - 1) / threadsX, (camera.getFrameBufferHeight() + threadsY - 1) / threadsY);

    // Allocate and initialize pixel random states on device
    Cuda::SmartPointer<curandState> d_pixelRandStates(framebufferSize, false);
    Cuda::initPixelCurandState<<<gridDimensions, blockDimensions>>>(d_pixelRandStates.get(), d_camera.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::clog << "Grid Dimensions: (" << gridDimensions.x << ", " << gridDimensions.y << ") " << "Block Dimensions: (" << blockDimensions.x << ", " << blockDimensions.y << ")\n";

    // Render
    Cuda::render<<<gridDimensions, blockDimensions>>>(d_world.get(), d_frameBuffer.get(), d_camera.get(), d_pixelRandStates.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Write output
    std::cout << "P3\n" << camera.imageWidth << " " << camera.getFrameBufferHeight() << "\n255\n";
    for(int i = 0; i < framebufferSize; i++) {
        write_color(std::cout, d_frameBuffer[i]);
    }

    // Free device memory
    Cuda::deleteScene<<<1, 1>>>(d_world.get());

    return 0;
}