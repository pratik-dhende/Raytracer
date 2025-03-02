#pragma once

#include "Raytracer.h"
#include "Hittable.h"

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
        hittables.clear();
    }

    void add(const std::shared_ptr<Hittable>& hittable) {
        hittables.emplace_back(hittable);
    }

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        double closestT = rayTInterval.max;
        HitInfo tempHitInfo;
        bool anyHit = false;

        for(const auto& hittable : hittables) {
            if (hittable->hit(ray, Interval(rayTInterval.min, closestT), tempHitInfo)) {
                closestT = tempHitInfo.t;
                hitInfo = tempHitInfo;
                anyHit = true;
            }
        }

        return anyHit;
    }

private:
    std::vector<std::shared_ptr<Hittable>> hittables;
};