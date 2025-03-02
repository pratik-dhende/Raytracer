#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"

#include <iostream>

int main() {
    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    Scene world;
    auto groundMaterial = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto centerMaterial = std::make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto leftMaterial   = std::make_shared<Dielectric>(1.0 / 1.33);
    auto rightMaterial  = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

    world.add(std::make_shared<Sphere>(Point3( 0.0, -100.5, -1.0), 100.0, groundMaterial));
    world.add(std::make_shared<Sphere>(Point3( 0.0,    0.0, -1.2),   0.5, centerMaterial));
    world.add(std::make_shared<Sphere>(Point3(-1.0,    0.0, -1.0),   0.5, leftMaterial));
    world.add(std::make_shared<Sphere>(Point3( 1.0,    0.0, -1.0),   0.5, rightMaterial));

    camera.render(world);

    return 0;
}