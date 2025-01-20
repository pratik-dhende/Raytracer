#include "Color.h"
#include "Ray.h"

#include <iostream>

Color getRayColor(const Ray& ray) {
    Vec3 unitDirection = unit_vector(ray.direction());
    auto a = 0.5 * (unitDirection.y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}

int main() {
    float aspectRatio = 16.0f / 9.0f;

    const int imageWidth = 400;
    const int imageHeight = std::max(1, static_cast<int>(imageWidth / aspectRatio));
    
    const float viewPortHeight = 2.0f;
    const float viewPortWidth = viewPortHeight * (static_cast<float>(imageWidth) / imageHeight);

    const float focalLength = 1.0f;
    const Point3 cameraPosition = Point3(0.0f, 0.0f, 0.0f);

    const Point3 pixel00Position = cameraPosition - Point3(viewPortWidth * 0.5f, -viewPortHeight * 0.5f, focalLength);

    const Vec3 du = Vec3(viewPortWidth / imageWidth, 0.0f, 0.0f);
    const Vec3 dv = Vec3(0.0f, -viewPortHeight / imageHeight, 0.0f);

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for(int i = 0; i < imageHeight; i++) {
        std::clog << "\rScanlines remaining: " << imageHeight - i << " " << std::flush; 
        for(int j = 0; j < imageWidth; j++) {
            const Point3 pixelPosition = pixel00Position + (j * du) + (i * dv);

            const Ray ray = Ray(pixelPosition, pixelPosition - cameraPosition);
            const Color rayColor = getRayColor(ray);

            write_color(std::cout, rayColor);
        }
    }

    std::clog << "\rDone.                 \n";

    return 0;
}