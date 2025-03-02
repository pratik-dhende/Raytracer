#pragma once

#include "Hittable.h"
#include "Color.h"
#include "Sphere.h"
#include "Scene.h"

class Camera {
public:
    double aspectRatio = 1.0;
    int imageWidth = 100;
    int samplesPerPixel = 10;

public:
    void render(const Hittable& world)  {
        init();

        std::cout << "P3\n" << this->imageWidth << " " << this->imageHeight << "\n255\n";

        for(int i = 0; i < this->imageHeight; i++) {
            std::clog << "\rScanlines remaining: " << this->imageHeight - i << " " << std::flush; 
            for(int j = 0; j < this->imageWidth; j++) {
                Color pixelColor = Color(0.0);

                for(int sample = 0; sample < samplesPerPixel; ++sample) {
                    pixelColor += rayColor(sampleRay(j, i), world);
                }
    
                write_color(std::cout, pixelColor * this->pixelsPerSample);
            }
        }
    
        std::clog << "\rDone.                 \n";
    }

private:
    int imageHeight;
    double pixelsPerSample;

    Point3 pixel00Position;
    Point3 cameraPosition;

    Vec3 du;
    Vec3 dv;

    void init() {
        constexpr double viewPortHeight = 2.0;
        constexpr double focalLength = 1.0;

        this->imageHeight = std::max(1, static_cast<int>(imageWidth / aspectRatio));
        this->pixelsPerSample = 1.0 / static_cast<double>(samplesPerPixel);
        this->cameraPosition = Point3(0.0, 0.0, 0.0);
        
        const double viewPortWidth = viewPortHeight * (static_cast<double>(imageWidth) / static_cast<double>(imageHeight));
    
        Scene world;
        world.add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5));
        world.add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0));
    
        const Vec3 viewportU = Vec3(viewPortWidth, 0.0, 0.0);
        const Vec3 viewportV = Vec3(0.0, -viewPortHeight, 0.0);
    
        this->du = viewportU / static_cast<double>(imageWidth);
        this->dv = viewportV / static_cast<double>(imageHeight);
    
        const Point3 viewportUpperLeft = cameraPosition - Vec3(0.0, 0.0, focalLength) - viewportU / 2.0 - viewportV / 2.0;
        this->pixel00Position = viewportUpperLeft + 0.5 * (du + dv);
    }

    Color rayColor(const Ray& ray, const Hittable& world) const {
        HitInfo hitInfo;

        if (world.hit(ray, Interval(0.0, POSITIVE_INFINITY), hitInfo)) {
            auto reflectedRay = Vec3::randomUnitHemisphere(hitInfo.getNormal());
            return 0.5 * rayColor(Ray(hitInfo.p, reflectedRay), world);
            // return 0.5 * hitInfo.getNormal() + 0.5;
        }
        
        Vec3 unitDirection = ray.direction().normalized();
        auto a = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0);
    }

    Ray sampleRay(const int x, const int y) const {
        const Point3 offset = sampleSquare();
        const Point3 samplePosition = pixel00Position + ((static_cast<double>(x) + offset.x()) * du) + ((static_cast<double>(y) + offset.y())  * dv);
    
        return Ray(this->cameraPosition, samplePosition - this->cameraPosition);
    }

    Point3 sampleSquare() const {
        return Point3(random() - 0.5, random() - 0.5, 0.0);
    }
};