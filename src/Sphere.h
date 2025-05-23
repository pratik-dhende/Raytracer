#pragma once

#include "Hittable.h"
#include "Material.h"

#include <algorithm>
#include <memory>

class Sphere : public Hittable {
    public:
        Sphere(const Point3& _center, const double _radius, const std::shared_ptr<Material> material) : center(_center), radius(std::max(0.0, _radius)), material(material) {

        }

        bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
            Vec3 centerRayOrigin = center - ray.origin();

            double a = ray.direction().magnitudeSquared();
            double h = Vec3::dot(ray.direction(), centerRayOrigin);
            double c = centerRayOrigin.magnitudeSquared() - radius * radius;

            double discriminant = h * h - a * c;

            if (discriminant < 0.0)
                return false;
            
            double sqrtDiscriminant = std::sqrt(discriminant);
            double t1 = (h - sqrtDiscriminant) / a;
            double t2 = (h + sqrtDiscriminant) / a;

            if (!rayTInterval.surrounds(t1) && !rayTInterval.surrounds(t2)) {
                return false;
            }

            hitInfo.t = rayTInterval.surrounds(t1) ? t1 : t2;
            hitInfo.p = ray.at(hitInfo.t);
            hitInfo.setNormal(ray, (hitInfo.p - this->center) / this->radius);
            hitInfo.material = material;
            
            return true;
        }

    private:
        Point3 center;
        double radius;
        std::shared_ptr<Material> material;
};