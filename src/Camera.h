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
    int maxDepth = 10;

    double vertifcalFov = 90.0;
    Point3 eyePosition = Point3(0.0);
    Point3 lookAtPosition = Point3(0.0, 0.0, -1.0);
    Vec3 up = Point3(0.0, 1.0, 0.0);

    double defocusAngle = 0.0;
    double focusDistance = 10.0;

public:
    void render(const Hittable& world)  {
        init();

        std::cout << "P3\n" << this->imageWidth << " " << this->imageHeight << "\n255\n";

        for(int i = 0; i < this->imageHeight; i++) {
            std::clog << "\rScanlines remaining: " << this->imageHeight - i << " " << std::flush; 
            for(int j = 0; j < this->imageWidth; j++) {
                Color pixelColor = Color(0.0);

                for(int sample = 0; sample < samplesPerPixel; ++sample) {
                    pixelColor += rayColor(sampleRay(j, i), world, maxDepth);
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

    Vec3 du;
    Vec3 dv;

    Vec3 u;
    Vec3 v;
    Vec3 w;

    Vec3 defocusDiskU;
    Vec3 defocusDiskV;

    void init() {
        const double verticalFovRadians = degreesToRadians(vertifcalFov);
        
        const double viewPortHeight = 2.0 * std::tan(verticalFovRadians / 2.0) * focusDistance;
        this->imageHeight = std::max(1, static_cast<int>(imageWidth / aspectRatio));
        this->pixelsPerSample = 1.0 / static_cast<double>(samplesPerPixel);
        
        const double viewPortWidth = viewPortHeight * (static_cast<double>(imageWidth) / static_cast<double>(imageHeight));

        w = (eyePosition - lookAtPosition).normalized();
        u = Vec3::cross(up, w).normalized();
        v = Vec3::cross(w, u);
    
        const Vec3 viewportU = viewPortWidth * u;
        const Vec3 viewportV = viewPortHeight * -v;
    
        this->du = viewportU / static_cast<double>(imageWidth);
        this->dv = viewportV / static_cast<double>(imageHeight);
    
        const Point3 viewportUpperLeft = eyePosition - (focusDistance * w) - viewportU / 2.0 - viewportV / 2.0;
        this->pixel00Position = viewportUpperLeft + 0.5 * (du + dv);

        double defocusRadius = focusDistance * std::tan(degreesToRadians(defocusAngle / 2.0));
        defocusDiskU = defocusRadius * u;
        defocusDiskV = defocusRadius * v;
    }

    Color rayColor(const Ray& ray, const Hittable& world, int depth) const {
        if (depth == 0) {
            return Color(0.0);
        }

        HitInfo hitInfo;

        if (world.hit(ray, Interval(0.001, POSITIVE_INFINITY), hitInfo)) {
            Color attenuation;
            Ray scatteredRay;

            if (hitInfo.material->scatter(ray, hitInfo, attenuation, scatteredRay)) {
                return attenuation * rayColor(scatteredRay, world, depth - 1);
            }

            return Color(0.0);
        }
        
        Vec3 unitDirection = ray.direction().normalized();
        auto a = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0);
    }

    Ray sampleRay(const int x, const int y) const {
        const Point3 offset = sampleSquare();
        const Point3 samplePosition = pixel00Position + ((static_cast<double>(x) + offset.x()) * du) + ((static_cast<double>(y) + offset.y())  * dv);
        
        auto rayOrigin = defocusAngle <= 0.0 ? this->eyePosition : sampleDefocusDisk();
        auto rayTime = randomDouble();
        return Ray(rayOrigin, samplePosition - rayOrigin, rayTime);
    }

    Point3 sampleSquare() const {
        return Point3(randomDouble() - 0.5, randomDouble() - 0.5, 0.0);
    }

    Point3 sampleDefocusDisk() const {
        auto p = Vec3::randomVectorInUnitCircle();
        return this->eyePosition + (p.x() * this->defocusDiskU) + (p.y() * this->defocusDiskV);
    }
};