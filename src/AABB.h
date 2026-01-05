#pragma once

#include "Interval.h"
#include "Vec3.h"
#include "Ray.h"

class AABB {
public:
    AABB() noexcept {
        
    };

    AABB(const Point3& a, const Point3& b) noexcept {
        m_x = Interval(std::min(a.x(), b.x()), std::max(a.x(), b.x()));
        m_y = Interval(std::min(a.y(), b.y()), std::max(a.y(), b.y()));
        m_z = Interval(std::min(a.z(), b.z()), std::max(a.z(), b.z()));

        addMinPadding();
    }

    AABB(const Interval& x, const Interval& y, const Interval& z) noexcept : m_x(x), m_y(y), m_z(z) {
        addMinPadding();
    }

    AABB(const AABB& aabb0, const AABB& aabb1) noexcept {
        m_x = Interval(aabb0.m_x, aabb1.m_x);
        m_y = Interval(aabb0.m_y, aabb1.m_y);
        m_z = Interval(aabb0.m_z, aabb1.m_z);

        addMinPadding();
    }

    Interval x() const noexcept {
        return m_x;
    }

    Interval y() const noexcept {
        return m_y;
    }

    Interval z() const noexcept {
        return m_z;
    }

    Interval axisInterval(int axis) const noexcept {
        return (axis == 0) ? m_x : (axis == 1 ? m_y : m_z);
    }

    bool hit(const Ray& ray, Interval rayTInterval) const noexcept {
        for(int axis = 0; axis < 3; ++axis) {
            auto t0 = (axisInterval(axis).min - ray.origin()[axis]) / ray.direction()[axis];
            auto t1 = (axisInterval(axis).max - ray.origin()[axis]) / ray.direction()[axis];

            rayTInterval = Interval::intersection(rayTInterval, Interval(std::min(t0, t1), std::max(t0, t1)));      
            
            if (rayTInterval.empty()) {
                return false;
            }
        }
        return true;
    }

    int longestAxis() const {
        return m_y.size() > m_z.size() ? (m_x.size() > m_y.size() ?  0 : 1) : (m_x.size() > m_z.size() ? 0 : 2);
    }

    static const AABB EMPTY, UNIVERSE;

private:
    Interval m_x, m_y, m_z;

    void addMinPadding() {
        constexpr double PADDING = 0.0001;

        if (m_x.size() < PADDING) {
            m_x.expand(PADDING);
        }

        if (m_y.size() < PADDING) {
            m_y.expand(PADDING);
        }

        if (m_z.size() < PADDING) {
            m_z.expand(PADDING);
        }
    }
};

const AABB AABB::EMPTY = AABB(Interval::EMPTY, Interval::EMPTY, Interval::EMPTY);
const AABB AABB::UNIVERSE = AABB(Interval::UNIVERSE, Interval::UNIVERSE, Interval::UNIVERSE);

AABB operator+(const AABB& aabb, const Vec3& offset) {
    return AABB(aabb.x() + offset.x(), aabb.y() + offset.y(), aabb.z() + offset.z());
}

AABB operator+(const Vec3& offset, const AABB& aabb) {
    return aabb + offset;
}