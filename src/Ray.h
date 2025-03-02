#pragma once

#include "Vec3.h"

class Ray {

public:
    Ray() : m_origin(0.0), m_direction(0.0) {}

    Ray(const Point3 &origin, const Vec3 &direction) noexcept
        : m_origin(origin), m_direction(direction) {}

    const Point3& origin() const noexcept { return m_origin; }
    const Vec3& direction() const noexcept { return m_direction; }

    Point3 at(double t) const noexcept {
        return m_origin + t * m_direction;
    }

private:
    Point3 m_origin;
    Vec3 m_direction;
};