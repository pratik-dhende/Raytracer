#pragma once

#include "Ray.h"
#include "Interval.h"

class HitInfo {
public:
    Point3f p;
    float t ;

    HitInfo() : p(0.0f), normal(0.0f), t(-1.0f), front(false) {}

    void setNormal(const Ray& ray, const Vec3f& unitNormal) {
        this->front = dot(ray.direction(), unitNormal) < 0.0f;
        this->normal = this->front ? unitNormal : -unitNormal;
    }

    Vec3f getNormal() const {
        return normal;
    }

private:
    Vec3f normal;
    bool front;
};

class Hittable {
    public:
        virtual ~Hittable() = default;

        virtual bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const = 0;
};