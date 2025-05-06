#pragma once

#include "Ray.h"
#include "Interval.h"

class Material;

class HitInfo {
public:
    Point3 p;
    double t ;

    Material* material;

    __device__ HitInfo() : p(0.0), normal(0.0), t(-1.0), front(false) {}

    __device__ void setNormal(const Ray& ray, const Vec3& unitNormal) {
        this->front = Vec3::dot(ray.direction(), unitNormal) < 0.0;
        this->normal = this->front ? unitNormal : -unitNormal;
    }

    __device__ Vec3 getNormal() const {
        return normal;
    }

    __device__ bool getFront() const {
        return front;
    }

private:
    Vec3 normal;
    bool front;
};

class Hittable {
    public:
        __device__ virtual ~Hittable() {};

        __device__ virtual bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const = 0;
};