#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"

#include <iostream>

int main() {
    Scene world;

    auto groundMaterial = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(std::make_shared<Sphere>(Point3(0.0 ,-1000.0 ,0.0), 1000.0, groundMaterial));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            auto chosenMaterial = random();
            Point3 center(a + 0.9 * random(), 0.2, b + 0.9 * random());

            if ((center - Point3(4.0, 0.2, 0.0)).magnitude() > 0.9) {
                std::shared_ptr<Material> sphereMaterial;

                if (chosenMaterial < 0.8) {
                    // Diffuse
                    auto albedo = Color::random() * Color::random();
                    sphereMaterial = std::make_shared<Lambertian>(albedo);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                } 
                else if (chosenMaterial < 0.95) {
                    // Metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random(0, 0.5);
                    sphereMaterial = std::make_shared<Metal>(albedo, fuzz);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                } 
                else {
                    // Glass
                    sphereMaterial = std::make_shared<Dielectric>(1.5);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                }
            }
        }
    }

    auto material1 = std::make_shared<Dielectric>(1.5);
    world.add(std::make_shared<Sphere>(Point3(0.0, 1.0, 0.0), 1.0, material1));

    auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(std::make_shared<Sphere>(Point3(-4.0, 1.0, 0.0), 1.0, material2));

    auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<Sphere>(Point3(4.0, 1.0, 0.0), 1.0, material3));
    
    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 1200;
    camera.samplesPerPixel = 10;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(13.0, 2.0, 3.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.6;
    camera.focusDistance = 10.0;

    camera.render(world);

    return 0;
}