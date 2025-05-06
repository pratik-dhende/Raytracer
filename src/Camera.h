#pragma once

#include "Hittable.h"
#include "Color.h"
#include "Sphere.h"
#include "Scene.h"
#include "Utility.h"

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
    void init() {
        const double verticalFovRadians = degreesToRadians(vertifcalFov);
        
        const double viewPortHeight = 2.0 * tan(verticalFovRadians / 2.0) * focusDistance;
        this->imageHeight = Cuda::max(1, static_cast<int>(imageWidth / aspectRatio));
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
    
        double defocusRadius = focusDistance * tan(degreesToRadians(defocusAngle / 2.0));
        defocusDiskU = defocusRadius * u;
        defocusDiskV = defocusRadius * v;
    }

    __device__ Ray sampleRay(const int x, const int y, curandState& randState) const {
        const Point3 offset = sampleSquare(randState);
        const Point3 samplePosition = pixel00Position + ((static_cast<double>(x) + offset.x()) * du) + ((static_cast<double>(y) + offset.y())  * dv);
        
        auto rayOrigin = defocusAngle <= 0.0 ? this->eyePosition : sampleDefocusDisk(randState);
        return Ray(rayOrigin, samplePosition - rayOrigin);
    }

    __device__ Point3 sampleSquare(curandState& randState) const {
        return Point3(Cuda::random(randState) - 0.5, Cuda::random(randState) - 0.5, 0.0);
    }

    __device__ Point3 sampleDefocusDisk(curandState& randState) const {
        auto p = Vec3::randomVectorInUnitCircle(randState);
        return this->eyePosition + (p.x() * this->defocusDiskU) + (p.y() * this->defocusDiskV);
    }

    __host__ __device__ int getFrameBufferHeight() const {
        return imageHeight;
    }

    __device__ double getPixelsPerSample() const {
        return pixelsPerSample;
    }

    ~Camera() {

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
};