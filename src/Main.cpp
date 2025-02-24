#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"

#include <iostream>

int main() {
    Camera camera;

    camera.aspectRatio = 16.0f / 9.0f;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;

    Scene world;
    world.add(std::make_shared<Sphere>(Point3f(0.0f, 0.0f, -1.0f), 0.5f));
    world.add(std::make_shared<Sphere>(Point3f(0.0f, -100.5f, -1.0f), 100.0f));

    camera.render(world);

    return 0;
}