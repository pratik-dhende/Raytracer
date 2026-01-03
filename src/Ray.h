#pragma once

#include "Vec3.h"

class Ray {

public:
    Ray() : m_origin(0.0), m_direction(0.0), m_time(0.0) {}

    Ray(const Point3 &origin, const Vec3 &direction) noexcept
        : Ray(origin, direction, 0.0) {}

    Ray(const Point3 &origin, const Vec3 &direction, const double time) noexcept
        : m_origin(origin), m_direction(direction), m_time(time) {}

    const Point3& origin() const noexcept { return m_origin; }
    const Vec3& direction() const noexcept { return m_direction; }
    const double time() const noexcept { return m_time; }

    Point3 at(double t) const noexcept {
        return m_origin + t * m_direction;
    }

private:
    Point3 m_origin;
    Vec3 m_direction;
    double m_time;
};