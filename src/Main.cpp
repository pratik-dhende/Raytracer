#include "Raytracer.h"
#include "Scene.h"
#include "Hittable.h"
#include "Sphere.h"

#include <iostream>

Color getRayColor(const Ray& ray, const Hittable& hittable) {
    HitInfo hitInfo;

    if (hittable.hit(ray, Interval(0.0f, F_INFINITY), hitInfo)) {
        return Color(hitInfo.getNormal()) * 0.5f + 0.5f;
    }
    
    Vec3f unitDirection = ray.direction().normalized();
    auto a = 0.5f * (unitDirection.y() + 1.0f);
    return (1.0f - a) * Color(1.0f) + a * Color(0.5f, 0.7f, 1.0f);
}

int main() {
    float aspectRatio = 16.0f / 9.0f;

    const int imageWidth = 400;
    const int imageHeight = std::max(1, static_cast<int>(imageWidth / aspectRatio));
    
    const float viewPortHeight = 2.0f;
    const float viewPortWidth = viewPortHeight * (static_cast<float>(imageWidth) / static_cast<float>(imageHeight));

    const float focalLength = 1.0f;
    const Point3f cameraPosition = Point3f(0.0f, 0.0f, 0.0f);

    Scene world;
    world.add(std::make_shared<Sphere>(Point3f(0.0f, 0.0f, -1.0f), 0.5f));
    world.add(std::make_shared<Sphere>(Point3f(0.0f, -100.5f, -1.0f), 100.0f));

    const Vec3f viewportU = Vec3f(viewPortWidth, 0.0f, 0.0f);
    const Vec3f viewportV = Vec3f(0.0f, -viewPortHeight, 0.0f);

    const Vec3f du = viewportU / static_cast<float>(imageWidth);
    const Vec3f dv = viewportV / static_cast<float>(imageHeight);

    const Point3f viewportUpperLeft = cameraPosition - Vec3f(0.0f, 0.0f, focalLength) - viewportU / 2.0f - viewportV / 2.0f;
    const Point3f pixel00Position = viewportUpperLeft + 0.5f * (du + dv);

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for(int i = 0; i < imageHeight; i++) {
        std::clog << "\rScanlines remaining: " << imageHeight - i << " " << std::flush; 
        for(int j = 0; j < imageWidth; j++) {
            const Point3f pixelPosition = pixel00Position + (static_cast<float>(j) * du) + (static_cast<float>(i)  * dv);

            const Ray ray = Ray(cameraPosition, pixelPosition - cameraPosition);
            const Color rayColor = getRayColor(ray, world);

            write_color(std::cout, rayColor);
        }
    }

    std::clog << "\rDone.                 \n";

    return 0;
}