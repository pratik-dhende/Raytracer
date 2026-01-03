#pragma once

#include "Interval.h"
#include "Vec3.h"
#include "Ray.h"

class AABB {
public:
    constexpr AABB() noexcept {};

    constexpr AABB(const Point3& min, const Point3& max) noexcept : x(min.x(), max.x()), y(min.y(), max.y()), z(min.z(), max.z()) {
        
    }

    constexpr AABB(const AABB& aabb0, const AABB& aabb1) noexcept {
        x = Interval(aabb0.x, aabb1.x);
        y = Interval(aabb0.y, aabb1.y);
        z = Interval(aabb0.z, aabb1.z);
    }

    Interval axisInterval(int axis) const noexcept {
        return (axis == 0) ? x : (axis == 1 ? y : z);
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

private:
    Interval x, y, z;
};