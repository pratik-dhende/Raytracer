#include "Vec3f.h"

class Ray {

public:
    Ray(const Point3f &origin, const Vec3f &direction) noexcept
        : m_origin(origin), m_direction(direction) {}

    const Point3f& origin() const noexcept { return m_origin; }
    const Vec3f& direction() const noexcept { return m_direction; }

    Point3f at(float t) const noexcept {
        return m_origin + t * m_direction;
    }

private:
    Point3f m_origin;
    Vec3f m_direction;
};