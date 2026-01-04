#pragma once

#include "Vec3.h"
#include "Hittable.h"

#include <iostream>

class Quad : public Hittable {
public:
    Quad(const Point3& bottemLeft, const Vec3& u, const Vec3& v, std::shared_ptr<Material> material) : q(bottemLeft), u(u), v(v), material(std::move(material)){
        m_aabb = AABB(AABB(q, q + u + v), AABB(q + u, q + v));

        n = Vec3::cross(u, v);
        d = Vec3::dot(n.normalized(), q);
        w = n / Vec3::dot(n, n);
    }

    bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        Vec3 normal = n.normalized();

        double denom = Vec3::dot(normal, ray.direction());

        if (std::abs(denom) < 0.0001) {
            return false;
        }

        double t = (d - Vec3::dot(normal, ray.origin())) / denom;

        if (!rayTInterval.contains(t)) {
            return false;
        }

        Vec3 p = ray.at(t) - q;
        
        double alpha = Vec3::dot(w, Vec3::cross(p, v));
        double beta = Vec3::dot(w, Vec3::cross(u, p));

        if (!isInterior(alpha, beta, hitInfo)) {
            return false;
        }   

        hitInfo.t = t;
        hitInfo.p = ray.at(hitInfo.t);
        hitInfo.setNormal(ray, n.normalized());
        hitInfo.material = material;

        return true;
    }

    AABB boundingVolume() const override {
        return m_aabb;
    }



private:
    AABB m_aabb;
    Point3 q;
    Vec3 u, v;
    std::shared_ptr<Material> material;
    Vec3 w;
    Vec3 n;
    double d;

private:
    virtual bool isInterior(const double a, const double b, HitInfo& hitInfo) const {
        Interval quadInterval(0.0, 1.0);

        if (!quadInterval.contains(a) || !quadInterval.contains(b)) {
            return false;
        }

        hitInfo.u = a;
        hitInfo.v = b;

        return true;
    }
};