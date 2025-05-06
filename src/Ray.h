#pragma once

#include "Vec3.h"

class Ray {

public:
    __device__ Ray() : m_origin(0.0), m_direction(0.0) {}

    __device__ Ray(const Point3 &origin, const Vec3 &direction) noexcept
        : m_origin(origin), m_direction(direction) {}

    __device__ const Point3& origin() const noexcept { return m_origin; }
    __device__ const Vec3& direction() const noexcept { return m_direction; }

    __device__ Point3 at(double t) const noexcept {
        return m_origin + t * m_direction;
    }

private:
    Point3 m_origin;
    Vec3 m_direction;
};