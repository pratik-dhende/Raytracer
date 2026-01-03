#pragma once

#include "Hittable.h"
#include "Scene.h"

class BVH : public Hittable{
public:
    BVH(std::vector<std::shared_ptr<Hittable>> hittables) : BVH(hittables, 0, static_cast<int>(hittables.size())) {

    }

    BVH(std::vector<std::shared_ptr<Hittable>>& hittables, const int start, const int end) {
        int n = end - start;
        if (n == 1) {
            left = right = hittables[start];
        }
        else if (n == 2) {
            left = hittables[start];
            right = hittables[start + 1];
        }
        else {
            int axis = randomInt(0, 2);

            const auto aabbComparator = [axis](const std::shared_ptr<Hittable>& first, const std::shared_ptr<Hittable>& second) {
                return first->boundingVolume().axisInterval(axis) < second->boundingVolume().axisInterval(axis);
            };

            int mid = start + n / 2;
            std::sort(hittables.begin() + start, hittables.begin() + end, aabbComparator);

            left = std::make_shared<BVH>(hittables, start, mid);
            right = std::make_shared<BVH>(hittables, mid, end);
        }

        aabb = AABB(left->boundingVolume(), right->boundingVolume());
    }

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const {
        if (!aabb.hit(ray, rayTInterval)) {
            return false;
        }

        bool leftHit = left->hit(ray, rayTInterval, hitInfo);
        bool rightHit = right->hit(ray, rayTInterval, hitInfo);

        return leftHit || rightHit;
    }

    AABB boundingVolume() const {
        return aabb;
    }

private:
    AABB aabb;
    std::shared_ptr<Hittable> left, right;
};