#include "Color.h"
#include "Ray.h"
#include "Vec3f.h"

#include <iostream>

double hitSphere(const Point3f &center, const double radius, const Ray& ray) {
    Vec3f centerRayOrigin = center - ray.origin();

    float a = ray.direction().magnitudeSquared();
    float h = dot(ray.direction(), centerRayOrigin);
    float c = centerRayOrigin.magnitudeSquared() - radius * radius;

    float discriminant = h * h - a * c;

    if (discriminant < 0.0f)
        return -1.0;
    
    return (h - std::sqrt(discriminant)) / a;
}

Color getRayColor(const Ray& ray) {
    auto sphereCenter = Point3f(0.0f, 0.0f, -5.0f);
    auto t = hitSphere(sphereCenter, 3.0f, ray);
    if (t > 0.0f) {
        const Vec3f& normal = (ray.at(t) - sphereCenter).normalized();
        return Color(normal) * 0.5f + 0.5f;
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
    const float viewPortWidth = viewPortHeight * (static_cast<float>(imageWidth) / imageHeight);

    const float focalLength = 1.0f;
    const Point3f cameraPosition = Point3f(0.0f, 0.0f, 0.0f);

    const Point3f pixel00Position = cameraPosition - Point3f(viewPortWidth * 0.5f, -viewPortHeight * 0.5f, focalLength);

    const Vec3f du = Vec3f(viewPortWidth / imageWidth, 0.0f, 0.0f);
    const Vec3f dv = Vec3f(0.0f, -viewPortHeight / imageHeight, 0.0f);

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for(int i = 0; i < imageHeight; i++) {
        std::clog << "\rScanlines remaining: " << imageHeight - i << " " << std::flush; 
        for(int j = 0; j < imageWidth; j++) {
            const Point3f pixelPosition = pixel00Position + (static_cast<float>(j) * du) + (static_cast<float>(i)  * dv);

            const Ray ray = Ray(pixelPosition, pixelPosition - cameraPosition);
            const Color rayColor = getRayColor(ray);

            write_color(std::cout, rayColor);
        }
    }

    std::clog << "\rDone.                 \n";

    return 0;
}