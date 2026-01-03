#pragma once

#include "Raytracer.h"
#include "Hittable.h"
#include "AABB.h"

#include <vector>
#include <memory>

class Scene : public Hittable {
public:
    Scene() {

    }

    Scene(const std::shared_ptr<Hittable>& hittable) {
        add(hittable);
    }

    void clear() {
        m_hittables.clear();
    }

    void add(const std::shared_ptr<Hittable>& hittable) {
        aabb = AABB(aabb, hittable->boundingVolume());
        m_hittables.emplace_back(hittable);
    }

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        double closestT = rayTInterval.max;
        HitInfo tempHitInfo;
        bool anyHit = false;

        for(const auto& hittable : m_hittables) {
            if (hittable->hit(ray, Interval(rayTInterval.min, closestT), tempHitInfo)) {
                closestT = tempHitInfo.t;
                hitInfo = tempHitInfo;
                anyHit = true;
            }
        }

        return anyHit;
    }

    AABB boundingVolume() const override {
        return aabb;
    }

    const std::vector<std::shared_ptr<Hittable>>& hittables() const noexcept{
        return m_hittables;
    } 

private:
    std::vector<std::shared_ptr<Hittable>> m_hittables;
    AABB aabb;
};