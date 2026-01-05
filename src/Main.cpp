#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "BVH.h"
#include "Texture.h"
#include "Quad.h"

#include <iostream>

void renderBouncingSpheres() {
    Scene scene;

    auto checkerTexture = std::make_shared<CheckerTexture>(0.32, Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
    scene.add(std::make_shared<Sphere>(Point3(0.0 ,-1000.0 ,0.0), 1000.0, std::make_shared<Lambertian>(checkerTexture)));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            auto chosenMaterial = randomDouble();
            Point3 center(a + 0.9 * randomDouble(), 0.2, b + 0.9 * randomDouble());

            if ((center - Point3(4.0, 0.2, 0.0)).magnitude() > 0.9) {
                std::shared_ptr<Material> sphereMaterial;

                if (chosenMaterial < 0.8) {
                    // Diffuse
                    auto albedo = Color::randomDouble() * Color::randomDouble();
                    sphereMaterial = std::make_shared<Lambertian>(albedo);
                    auto center2 = center + Vec3(0, randomDouble(0, 0.5), 0);
                    scene.add(std::make_shared<Sphere>(center, center2, 0.2, sphereMaterial));
                } 
                else if (chosenMaterial < 0.95) {
                    // Metal
                    auto albedo = Color::randomDouble(0.5, 1);
                    auto fuzz = randomDouble(0, 0.5);
                    sphereMaterial = std::make_shared<Metal>(albedo, fuzz);
                    scene.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                } 
                else {
                    // Glass
                    sphereMaterial = std::make_shared<Dielectric>(1.5);
                    scene.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                }
            }
        }
    }

    auto material1 = std::make_shared<Dielectric>(1.5);
    scene.add(std::make_shared<Sphere>(Point3(0.0, 1.0, 0.0), 1.0, material1));

    auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    scene.add(std::make_shared<Sphere>(Point3(-4.0, 1.0, 0.0), 1.0, material2));

    auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    scene.add(std::make_shared<Sphere>(Point3(4.0, 1.0, 0.0), 1.0, material3));
    
    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(13.0, 2.0, 3.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.6;
    camera.focusDistance = 10.0;

    camera.backgroundColor = Color(0.70, 0.80, 1.00);

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderCheckeredSpheres() {
    Scene scene;

    auto checkerTexture = std::make_shared<CheckerTexture>(0.32, Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));

    scene.add(std::make_shared<Sphere>(Point3(0,-10, 0), 10, std::make_shared<Lambertian>(checkerTexture)));
    scene.add(std::make_shared<Sphere>(Point3(0, 10, 0), 10, std::make_shared<Lambertian>(checkerTexture)));

    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(13.0, 2.0, 3.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.0;
    camera.focusDistance = 10.0;

    camera.backgroundColor = Color(0.70, 0.80, 1.00);

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderEarth() {
    Scene scene;

    auto earthTexture = std::make_shared<ImageTexture>("../textures/earthmap.jpg");
    scene.add(std::make_shared<Sphere>(Point3(0.0), 2.0, std::make_shared<Lambertian>(earthTexture)));

    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(0.0, 0.0, 12.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.0;
    camera.focusDistance = 10.0;

    camera.backgroundColor = Color(0.70, 0.80, 1.00);

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderPerlinSpheres() {
    Scene scene;
    
    auto perlinNoiseTexture = std::make_shared<PerlinNoiseTexture>(4.0);
    scene.add(std::make_shared<Sphere>(Point3(0.0, -1000.0, 0.0), 1000.0, std::make_shared<Lambertian>(perlinNoiseTexture)));
    scene.add(std::make_shared<Sphere>(Point3(0.0, 2.0, 0.0), 2.0, std::make_shared<Lambertian>(perlinNoiseTexture)));

    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    camera.vertifcalFov = 20.0;
    camera.eyePosition = Point3(13.0, 2.0, 3.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0.0;
    camera.focusDistance = 10.0;

    camera.backgroundColor = Color(0.70, 0.80, 1.00);

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderQuads() {
    Scene scene;

    // Materials
    auto left_red     = std::make_shared<Lambertian>(Color(1.0, 0.2, 0.2));
    auto back_green   = std::make_shared<Lambertian>(Color(0.2, 1.0, 0.2));
    auto right_blue   = std::make_shared<Lambertian>(Color(0.2, 0.2, 1.0));
    auto upper_orange = std::make_shared<Lambertian>(Color(1.0, 0.5, 0.0));
    auto lower_teal   = std::make_shared<Lambertian>(Color(0.2, 0.8, 0.8));

    // Quads
    scene.add(std::make_shared<Quad>(Point3(-3.0, -2.0, 5.0), Vec3(0.0, 0.0,-4.0), Vec3(0.0, 4.0, 0.0), left_red));
    scene.add(std::make_shared<Quad>(Point3(-2.0, -2.0, 0.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), back_green));
    scene.add(std::make_shared<Quad>(Point3( 3.0, -2.0, 1.0), Vec3(0.0, 0.0, 4.0), Vec3(0.0, 4.0, 0.0), right_blue));
    scene.add(std::make_shared<Quad>(Point3(-2.0,  3.0, 1.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.0), upper_orange));
    scene.add(std::make_shared<Quad>(Point3(-2.0, -3.0, 5.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0,-4.0), lower_teal));

    Camera camera;

    camera.aspectRatio      = 1.0;
    camera.imageWidth       = 400;
    camera.samplesPerPixel  = 100;
    camera.maxDepth         = 50;

    camera.vertifcalFov     = 80.0;
    camera.eyePosition      = Point3(0.0, 0.0, 9.0);
    camera.lookAtPosition   = Point3(0.0, 0.0, 0.0);
    camera.up               = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0;

    camera.backgroundColor = Color(0.70, 0.80, 1.00);

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderSimpleLight() {
    Scene scene;

    auto perlinNoiseTexture = std::make_shared<PerlinNoiseTexture>(4);
    scene.add(std::make_shared<Sphere>(Point3(0.0, -1000.0, 0.0), 1000.0, std::make_shared<Lambertian>(perlinNoiseTexture)));
    scene.add(std::make_shared<Sphere>(Point3(0.0, 2.0, 0.0), 2.0, std::make_shared<Lambertian>(perlinNoiseTexture)));

    auto diffuseLightMaterial = std::make_shared<DiffuseLight>(Color(4.0));
    scene.add(std::make_shared<Quad>(Point3(3.0, 1.0, -2.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), diffuseLightMaterial));

    Camera camera;

    camera.aspectRatio      = 16.0 / 9.0;
    camera.imageWidth       = 400;
    camera.samplesPerPixel  = 100;
    camera.maxDepth         = 50;
    camera.backgroundColor  = Color(0.0);

    camera.vertifcalFov     = 20;
    camera.eyePosition      = Point3(26.0, 3.0, 6.0);
    camera.lookAtPosition   = Point3(0.0, 2.0, 0.0);
    camera.up               = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0;

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}


int main() {
    switch(6) {
        case 1 : renderBouncingSpheres(); break;
        case 2 : renderCheckeredSpheres(); break;
        case 3 : renderEarth(); break;
        case 4 : renderPerlinSpheres(); break;
        case 5 : renderQuads(); break;
        case 6 : renderSimpleLight(); break;
    }

    return 0;
}