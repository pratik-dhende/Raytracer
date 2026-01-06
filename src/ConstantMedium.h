#pragma once

#include "Hittable.h"
#include "Material.h"
#include "Texture.h"

class ConstantMedium : public Hittable {
  public:
    ConstantMedium(std::shared_ptr<Hittable> boundary, double density, std::shared_ptr<Texture> tex)
      : m_boundary(boundary), negativeInverseDensity(-1 / density),
        phaseFunction(std::make_shared<Isotropic>(tex))
    {}

    ConstantMedium(std::shared_ptr<Hittable> boundary, double density, const Color& albedo)
      : m_boundary(boundary), negativeInverseDensity(-1 / density),
        phaseFunction(std::make_shared<Isotropic>(albedo))
    {}

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        HitInfo entryHitInfo, exitHitInfo;

        if (!m_boundary->hit(ray, Interval::UNIVERSE, entryHitInfo))
            return false;

        if (!m_boundary->hit(ray, Interval(entryHitInfo.t+0.0001, INFINITY), exitHitInfo))
            return false;

        if (entryHitInfo.t < rayTInterval.min) entryHitInfo.t = rayTInterval.min;
        if (exitHitInfo.t > rayTInterval.max) exitHitInfo.t = rayTInterval.max;

        if (entryHitInfo.t >= exitHitInfo.t)
            return false;

        if (entryHitInfo.t < 0)
            entryHitInfo.t = 0;

        auto ray_length = ray.direction().magnitude();
        auto distance_inside_boundary = (exitHitInfo.t - entryHitInfo.t) * ray_length;
        auto hit_distance = negativeInverseDensity * std::log(randomDouble());

        if (hit_distance > distance_inside_boundary)
            return false;

        hitInfo.t = entryHitInfo.t + hit_distance / ray_length;
        hitInfo.p = ray.at(hitInfo.t);
        hitInfo.material = phaseFunction;

        return true;
    }

    AABB boundingVolume() const override { return m_boundary->boundingVolume(); }

  private:
    std::shared_ptr<Hittable> m_boundary;
    double negativeInverseDensity;
    std::shared_ptr<Material> phaseFunction;
};