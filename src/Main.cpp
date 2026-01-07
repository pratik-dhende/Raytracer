#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "BVH.h"
#include "Texture.h"
#include "Quad.h"
#include "ConstantMedium.h"

#include <iostream>

void renderBouncingSpheres() {
    Scene scene;

    auto checkerTexture = std::make_shared<CheckerTexture>(0.32, Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
    scene.add(std::make_shared<Sphere>(Point3(0.0, -1000.0, 0.0), 1000.0, std::make_shared<Lambertian>(checkerTexture)));

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

    camera.verticalFov = 20.0;
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

    camera.verticalFov = 20.0;
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

    camera.verticalFov = 20.0;
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

    camera.verticalFov = 20.0;
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
    auto left_red = std::make_shared<Lambertian>(Color(1.0, 0.2, 0.2));
    auto back_green = std::make_shared<Lambertian>(Color(0.2, 1.0, 0.2));
    auto right_blue = std::make_shared<Lambertian>(Color(0.2, 0.2, 1.0));
    auto upper_orange = std::make_shared<Lambertian>(Color(1.0, 0.5, 0.0));
    auto lower_teal = std::make_shared<Lambertian>(Color(0.2, 0.8, 0.8));

    // Quads
    scene.add(std::make_shared<Quad>(Point3(-3.0, -2.0, 5.0), Vec3(0.0, 0.0,-4.0), Vec3(0.0, 4.0, 0.0), left_red));
    scene.add(std::make_shared<Quad>(Point3(-2.0, -2.0, 0.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), back_green));
    scene.add(std::make_shared<Quad>(Point3( 3.0, -2.0, 1.0), Vec3(0.0, 0.0, 4.0), Vec3(0.0, 4.0, 0.0), right_blue));
    scene.add(std::make_shared<Quad>(Point3(-2.0,  3.0, 1.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.0), upper_orange));
    scene.add(std::make_shared<Quad>(Point3(-2.0, -3.0, 5.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0,-4.0), lower_teal));

    Camera camera;

    camera.aspectRatio = 1.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;

    camera.verticalFov = 80.0;
    camera.eyePosition = Point3(0.0, 0.0, 9.0);
    camera.lookAtPosition = Point3(0.0, 0.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

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
    scene.add(std::make_shared<Sphere>(Point3(0.0, 7.0, 0.0), 2.0, diffuseLightMaterial));
    scene.add(std::make_shared<Quad>(Point3(3.0, 1.0, -2.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), diffuseLightMaterial));

    Camera camera;

    camera.aspectRatio = 16.0 / 9.0;
    camera.imageWidth = 400;
    camera.samplesPerPixel = 100;
    camera.maxDepth = 50;
    camera.backgroundColor = Color(0.0);

    camera.verticalFov = 20;
    camera.eyePosition = Point3(26.0, 3.0, 6.0);
    camera.lookAtPosition = Point3(0.0, 2.0, 0.0);
    camera.up = Vec3(0.0, 1.0, 0.0);

    camera.defocusAngle = 0;

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderCornellBox() {
    Scene scene;

    auto red = std::make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = std::make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = std::make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = std::make_shared<DiffuseLight>(Color(15, 15, 15));

    scene.add(std::make_shared<Quad>(Point3(555, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), green));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), red));
    scene.add(std::make_shared<Quad>(Point3(343, 554, 332), Vec3(-130, 0, 0), Vec3(0, 0,-105), light));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
    scene.add(std::make_shared<Quad>(Point3(555, 555, 555), Vec3(-555, 0, 0), Vec3(0, 0,-555), white));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), white));

    std::shared_ptr<Hittable> box1 = Quad::box(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = std::make_shared<RotateY>(box1, 15);
    box1 = std::make_shared<Translate>(box1, Vec3(265, 0, 295));
    scene.add(box1);

    std::shared_ptr<Hittable> box2 = Quad::box(Point3(0, 0, 0), Point3(165, 165, 165), white);
    box2 = std::make_shared<RotateY>(box2, -18);
    box2 = std::make_shared<Translate>(box2, Vec3(130, 0, 65));
    scene.add(box2);

    Camera camera;

    camera.aspectRatio = 1.0;
    camera.imageWidth = 600;
    camera.samplesPerPixel = 200;
    camera.maxDepth = 50;
    camera.backgroundColor = Color(0, 0, 0);

    camera.verticalFov = 40;
    camera.eyePosition = Point3(278, 278, -800);
    camera.lookAtPosition = Point3(278, 278, 0);
    camera.up = Vec3(0, 1, 0);

    camera.defocusAngle = 0;

    std::shared_ptr<Hittable> world = std::make_shared<BVH>(scene.hittables());
    camera.render(*world);
}

void renderCornellSmoke() {
    Scene scene;

    auto red = std::make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = std::make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = std::make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = std::make_shared<DiffuseLight>(Color(7, 7, 7));

    scene.add(std::make_shared<Quad>(Point3(555, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), green));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), red));
    scene.add(std::make_shared<Quad>(Point3(113, 554, 127), Vec3(330, 0, 0), Vec3(0, 0, 305), light));
    scene.add(std::make_shared<Quad>(Point3(0, 555, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
    scene.add(std::make_shared<Quad>(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), white));

    std::shared_ptr<Hittable> box1 = Quad::box(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = std::make_shared<RotateY>(box1, 15);
    box1 = std::make_shared<Translate>(box1, Vec3(265, 0, 295));

    std::shared_ptr<Hittable> box2 = Quad::box(Point3(0, 0, 0), Point3(165, 165, 165), white);
    box2 = std::make_shared<RotateY>(box2, -18);
    box2 = std::make_shared<Translate>(box2, Vec3(130, 0, 65));

    scene.add(std::make_shared<ConstantMedium>(box1, 0.01, Color(0, 0, 0)));
    scene.add(std::make_shared<ConstantMedium>(box2, 0.01, Color(1, 1, 1)));

    Camera camera;

    camera.aspectRatio = 1.0;
    camera.imageWidth = 600;
    camera.samplesPerPixel = 200;
    camera.maxDepth = 50;
    camera.backgroundColor = Color(0, 0, 0);

    camera.verticalFov = 40;
    camera.eyePosition = Point3(278, 278, -800);
    camera.lookAtPosition = Point3(278, 278, 0);
    camera.up = Vec3(0, 1, 0);

    camera.defocusAngle = 0;

    camera.render(scene);
}

void renderFinalScene(int image_width, int samples_per_pixel, int max_depth) {
    Scene boxes1;
    auto ground = std::make_shared<Lambertian>(Color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = randomDouble(1, 101);
            auto z1 = z0 + w;

            boxes1.add(Quad::box(Point3(x0,y0,z0), Point3(x1,y1,z1), ground));
        }
    }

    Scene world;

    world.add(std::make_shared<BVH>(boxes1.hittables()));

    auto light = std::make_shared<DiffuseLight>(Color(7, 7, 7));
    world.add(std::make_shared<Quad>(Point3(123, 554, 147), Vec3(300, 0, 0), Vec3(0, 0, 265), light));

    auto center1 = Point3(400, 400, 200);
    auto center2 = center1 + Vec3(30, 0, 0);
    auto sphere_material = std::make_shared<Lambertian>(Color(0.7, 0.3, 0.1));
    world.add(std::make_shared<Sphere>(center1, center2, 50, sphere_material));

    world.add(std::make_shared<Sphere>(Point3(260, 150, 45), 50, std::make_shared<Dielectric>(1.5)));
    world.add(std::make_shared<Sphere>(
        Point3(0, 150, 145), 50, std::make_shared<Metal>(Color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = std::make_shared<Sphere>(Point3(360, 150, 145), 70, std::make_shared<Dielectric>(1.5));
    world.add(boundary);
    world.add(std::make_shared<ConstantMedium>(boundary, 0.2, Color(0.2, 0.4, 0.9)));
    boundary = std::make_shared<Sphere>(Point3(0, 0, 0), 5000, std::make_shared<Dielectric>(1.5));
    world.add(std::make_shared<ConstantMedium>(boundary, .0001, Color(1, 1, 1)));

    auto earthMapMaterial = std::make_shared<Lambertian>(std::make_shared<ImageTexture>("../textures/earthmap.jpg"));
    world.add(std::make_shared<Sphere>(Point3(400, 200, 400), 100, earthMapMaterial));
    auto perlinNoiseTexture = std::make_shared<PerlinNoiseTexture>(0.2);
    world.add(std::make_shared<Sphere>(Point3(220, 280, 300), 80, std::make_shared<Lambertian>(perlinNoiseTexture)));

    Scene boxes2;
    auto white = std::make_shared<Lambertian>(Color(.73, .73, .73));
    int totalSpheres = 1000;
    for (int j = 0; j < totalSpheres; j++) {
        boxes2.add(std::make_shared<Sphere>(Point3::randomDouble(0, 165), 10, white));
    }

    world.add(std::make_shared<Translate>(
        std::make_shared<RotateY>(
            std::make_shared<BVH>(boxes2.hittables()), 15),
            Vec3(-100, 270, 395)
        )
    );

    Camera cam;

    cam.aspectRatio = 1.0;
    cam.imageWidth = image_width;
    cam.samplesPerPixel = samples_per_pixel;
    cam.maxDepth = max_depth;
    cam.backgroundColor = Color(0, 0, 0);

    cam.verticalFov = 40;
    cam.eyePosition = Point3(478, 278, -600);
    cam.lookAtPosition = Point3(278, 278, 0);
    cam.up = Vec3(0, 1, 0);

    cam.defocusAngle = 0;

    cam.render(world);
}

int main() {
    switch(10) {
        case 1 : renderBouncingSpheres(); break;
        case 2 : renderCheckeredSpheres(); break;
        case 3 : renderEarth(); break;
        case 4 : renderPerlinSpheres(); break;
        case 5 : renderQuads(); break;
        case 6 : renderSimpleLight(); break;
        case 7 : renderCornellBox(); break;
        case 8 : renderCornellSmoke(); break;
        case 9:  renderFinalScene(800, 10000, 40); break;
        default: renderFinalScene(400, 250, 4); break;
    }

    return 0;
}