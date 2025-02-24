#pragma once

#include "Hittable.h"
#include "Color.h"
#include "Sphere.h"
#include "Scene.h"

class Camera {
public:
    float aspectRatio = 1.0f;
    int imageWidth = 100;

public:
    void render(const Hittable& world) {
        init();

        std::cout << "P3\n" << this->imageWidth << " " << this->imageHeight << "\n255\n";

        for(int i = 0; i < this->imageHeight; i++) {
            std::clog << "\rScanlines remaining: " << this->imageHeight - i << " " << std::flush; 
            for(int j = 0; j < this->imageWidth; j++) {
                const Point3f pixelPosition = pixel00Position + (static_cast<float>(j) * du) + (static_cast<float>(i)  * dv);
    
                const Ray ray = Ray(cameraPosition, pixelPosition - cameraPosition);
                const Color pixelColor = rayColor(ray, world);
    
                write_color(std::cout, pixelColor);
            }
        }
    
        std::clog << "\rDone.                 \n";
    }

private:
    int imageHeight;

    Point3f pixel00Position;
    Point3f cameraPosition;

    Vec3f du;
    Vec3f dv;

    void init() {
        this->imageHeight = std::max(1, static_cast<int>(imageWidth / aspectRatio));
        
        constexpr float viewPortHeight = 2.0f;
        const float viewPortWidth = viewPortHeight * (static_cast<float>(imageWidth) / static_cast<float>(imageHeight));
    
        constexpr float focalLength = 1.0f;
        this->cameraPosition = Point3f(0.0f, 0.0f, 0.0f);
    
        Scene world;
        world.add(std::make_shared<Sphere>(Point3f(0.0f, 0.0f, -1.0f), 0.5f));
        world.add(std::make_shared<Sphere>(Point3f(0.0f, -100.5f, -1.0f), 100.0f));
    
        const Vec3f viewportU = Vec3f(viewPortWidth, 0.0f, 0.0f);
        const Vec3f viewportV = Vec3f(0.0f, -viewPortHeight, 0.0f);
    
        this->du = viewportU / static_cast<float>(imageWidth);
        this->dv = viewportV / static_cast<float>(imageHeight);
    
        const Point3f viewportUpperLeft = cameraPosition - Vec3f(0.0f, 0.0f, focalLength) - viewportU / 2.0f - viewportV / 2.0f;
        this->pixel00Position = viewportUpperLeft + 0.5f * (du + dv);
    }

    Color rayColor(const Ray& ray, const Hittable& world) const {
        HitInfo hitInfo;

        if (world.hit(ray, Interval(0.0f, F_INFINITY), hitInfo)) {
            return Color(hitInfo.getNormal()) * 0.5f + 0.5f;
        }
        
        Vec3f unitDirection = ray.direction().normalized();
        auto a = 0.5f * (unitDirection.y() + 1.0f);
        return (1.0f - a) * Color(1.0f) + a * Color(0.5f, 0.7f, 1.0f);
    }
};