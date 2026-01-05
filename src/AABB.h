#pragma once

#include "Interval.h"
#include "Vec3.h"
#include "Ray.h"

class AABB {
public:
    AABB() noexcept {
        
    };

    AABB(const Point3& a, const Point3& b) noexcept {
        x = Interval(std::min(a.x(), b.x()), std::max(a.x(), b.x()));
        y = Interval(std::min(a.y(), b.y()), std::max(a.y(), b.y()));
        z = Interval(std::min(a.z(), b.z()), std::max(a.z(), b.z()));

        addMinPadding();
    }

    AABB(const Interval& _x, const Interval& _y, const Interval& _z) noexcept : x(_x), y(_y), z(_z) {
        addMinPadding();
    }

    AABB(const AABB& aabb0, const AABB& aabb1) noexcept {
        x = Interval(aabb0.x, aabb1.x);
        y = Interval(aabb0.y, aabb1.y);
        z = Interval(aabb0.z, aabb1.z);

        addMinPadding();
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

    int longestAxis() const {
        return y.size() > z.size() ? (x.size() > y.size() ?  0 : 1) : (x.size() > z.size() ? 0 : 2);
    }

    static const AABB EMPTY, UNIVERSE;

private:
    Interval x, y, z;

    void addMinPadding() {
        constexpr double PADDING = 0.0001;

        if (x.size() < PADDING) {
            x.expand(PADDING);
        }

        if (y.size() < PADDING) {
            y.expand(PADDING);
        }

        if (z.size() < PADDING) {
            z.expand(PADDING);
        }
    }
};

const AABB AABB::EMPTY = AABB(Interval::EMPTY, Interval::EMPTY, Interval::EMPTY);
const AABB AABB::UNIVERSE = AABB(Interval::UNIVERSE, Interval::UNIVERSE, Interval::UNIVERSE);