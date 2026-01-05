#pragma once

#include "Ray.h"
#include "Interval.h"
#include "AABB.h"

class Material;

class HitInfo {
public:
    Point3 p;
    double t ;
    std::shared_ptr<Material> material;
    double u;
    double v;
    Vec3 normal;

    HitInfo() : p(0.0), normal(0.0), t(-1.0), m_front(false) {}

    void setOutwardNormal(const Ray& ray, const Vec3& unitNormal) {
        m_front = Vec3::dot(ray.direction(), unitNormal) < 0.0;
        normal = m_front ? unitNormal : -unitNormal;
    }

    bool front() const {
        return m_front;
    }

private:
    bool m_front;
};

class Hittable {
    public:
        virtual ~Hittable() = default;

        virtual bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const = 0;

        virtual AABB boundingVolume() const = 0;
};

class Translate : public Hittable {
  public:
    Translate(std::shared_ptr<Hittable> hittable, const Vec3& offset) : m_hittable(hittable), m_offset(offset)
    {
        aabb = hittable->boundingVolume() + m_offset;
    }

    bool hit(const Ray& r, const Interval& ray_t, HitInfo& hitInfo) const override {
        Ray offsetRay(r.origin() - m_offset, r.direction(), r.time());
        
        if (!m_hittable->hit(offsetRay, ray_t, hitInfo))
            return false;

        hitInfo.p += m_offset;

        return true;
    }

    AABB boundingVolume() const override {
        return aabb;
    }

  private:
    std::shared_ptr<Hittable> m_hittable;
    Vec3 m_offset;
    AABB aabb;
};

class RotateY : public Hittable {
public:
    RotateY(std::shared_ptr<Hittable> hittable, double angle) : m_hittable(hittable) {
        auto radians = degreesToRadians(angle);

        sin_theta = std::sin(radians);
        cos_theta = std::cos(radians);

        aabb = hittable->boundingVolume();

        Point3 aabbMin(INFINITY);
        Point3 aabbMax(-INFINITY);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * aabb.x().max + (1 - i) * aabb.x().min;
                    auto y = j * aabb.y().max + (1 - j) * aabb.y().min;
                    auto z = k * aabb.z().max + (1 - k) * aabb.z().min;

                    Vec3 rotatedXYZ((cos_theta * x) + (sin_theta * z), y, (-sin_theta * x) + (cos_theta * z));

                    for (int c = 0; c < 3; c++) {
                        aabbMin[c] = std::min(aabbMin[c], rotatedXYZ[c]);
                        aabbMax[c] = std::max(aabbMax[c], rotatedXYZ[c]);
                    }
                }
            }
        }

        aabb = AABB(aabbMin, aabbMax);
    }

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        auto rotatedOrigin = Point3(
            (cos_theta * ray.origin().x()) - (sin_theta * ray.origin().z()),
            ray.origin().y(),
            (sin_theta * ray.origin().x()) + (cos_theta * ray.origin().z())
        );

        auto rotatedDirection = Vec3(
            (cos_theta * ray.direction().x()) - (sin_theta * ray.direction().z()),
            ray.direction().y(),
            (sin_theta * ray.direction().x()) + (cos_theta * ray.direction().z())
        );

        Ray rotatedRay(rotatedOrigin, rotatedDirection, ray.time());

        if (!m_hittable->hit(rotatedRay, rayTInterval, hitInfo))
            return false;

        hitInfo.p = Point3(
            (cos_theta * hitInfo.p.x()) + (sin_theta * hitInfo.p.z()),
            hitInfo.p.y(),
            (-sin_theta * hitInfo.p.x()) + (cos_theta * hitInfo.p.z())
        );

        hitInfo.normal = Vec3(
            (cos_theta * hitInfo.normal.x()) + (sin_theta * hitInfo.normal.z()),
            hitInfo.normal.y(),
            (-sin_theta * hitInfo.normal.x()) + (cos_theta * hitInfo.normal.z())
        );

        return true;
    }

    AABB boundingVolume() const override { return aabb; }

private:
    std::shared_ptr<Hittable> m_hittable;
    double sin_theta;
    double cos_theta;
    AABB aabb;
};
