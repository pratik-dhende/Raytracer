#pragma once

#include "Hittable.h"

#include <algorithm>

class Sphere : public Hittable {
    public:
        Sphere(const Point3f& _center, const float _radius) : center(_center), radius(std::max(0.0f, _radius)) {

        }

        bool hit(const Ray& ray, const float rayTMin, const float rayTMax, HitInfo& hitInfo) const override {
            Vec3f centerRayOrigin = center - ray.origin();

            float a = ray.direction().magnitudeSquared();
            float h = dot(ray.direction(), centerRayOrigin);
            float c = centerRayOrigin.magnitudeSquared() - radius * radius;

            float discriminant = h * h - a * c;

            if (discriminant < 0.0f)
                return false;
            
            float sqrtDiscriminant = std::sqrt(discriminant);
            float t1 = (h - sqrtDiscriminant) / a;
            float t2 = (h + sqrtDiscriminant) / a;

            if (t1 >= rayTMax || t2 <= rayTMin || t1 <= rayTMin && t2 >= rayTMax) {
                return false;
            }

            hitInfo.t = t1 > rayTMin ? t1 : t2;
            hitInfo.p = ray.at(hitInfo.t);
            hitInfo.setNormal(ray, (hitInfo.p - this->center) / this->radius);
            
            return true;
        }

    private:
        Point3f center;
        float radius;
};