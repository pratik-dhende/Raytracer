#pragma once

#include "Ray.h"
#include "Interval.h"
#include "AABB.h"

class Material;

class HitInfo {
public:
    Point3 p;
    double t ;
    std::shared_ptr<Material> material;

    HitInfo() : p(0.0), normal(0.0), t(-1.0), front(false) {}

    void setNormal(const Ray& ray, const Vec3& unitNormal) {
        this->front = Vec3::dot(ray.direction(), unitNormal) < 0.0;
        this->normal = this->front ? unitNormal : -unitNormal;
    }

    Vec3 getNormal() const {
        return normal;
    }

    bool getFront() const {
        return front;
    }

private:
    Vec3 normal;
    bool front;
};

class Hittable {
    public:
        virtual ~Hittable() = default;

        virtual bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const = 0;

        virtual AABB boundingVolume() const = 0;
};