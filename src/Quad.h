#pragma once

#include "Vec3.h"
#include "Hittable.h"

#include <iostream>

class Quad : public Hittable {
public:
    static std::shared_ptr<Hittable> box(const Point3& a, const Point3& b, std::shared_ptr<Material> material)
    {
        auto boxScene = std::make_shared<Scene>();

        auto boxMin = Point3(std::min(a.x(),b.x()), std::min(a.y(),b.y()), std::min(a.z(),b.z()));
        auto boxMax = Point3(std::max(a.x(),b.x()), std::max(a.y(),b.y()), std::max(a.z(),b.z()));

        auto dx = Vec3(boxMax.x() - boxMin.x(), 0, 0);
        auto dy = Vec3(0, boxMax.y() - boxMin.y(), 0);
        auto dz = Vec3(0, 0, boxMax.z() - boxMin.z());

        boxScene->add(std::make_shared<Quad>(Point3(boxMin.x(), boxMin.y(), boxMax.z()),  dx,  dy, material)); // front
        boxScene->add(std::make_shared<Quad>(Point3(boxMax.x(), boxMin.y(), boxMax.z()), -dz,  dy, material)); // right
        boxScene->add(std::make_shared<Quad>(Point3(boxMax.x(), boxMin.y(), boxMin.z()), -dx,  dy, material)); // back
        boxScene->add(std::make_shared<Quad>(Point3(boxMin.x(), boxMin.y(), boxMin.z()),  dz,  dy, material)); // left
        boxScene->add(std::make_shared<Quad>(Point3(boxMin.x(), boxMax.y(), boxMax.z()),  dx, -dz, material)); // top
        boxScene->add(std::make_shared<Quad>(Point3(boxMin.x(), boxMin.y(), boxMin.z()),  dx,  dz, material)); // bottom

        return boxScene;
    }

    Quad(const Point3& bottemLeft, const Vec3& u, const Vec3& v, std::shared_ptr<Material> material) : q(bottemLeft), u(u), v(v), m_material(std::move(material)){
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
        hitInfo.setOutwardNormal(ray, n.normalized());
        hitInfo.material = m_material;

        return true;
    }

    AABB boundingVolume() const override {
        return m_aabb;
    }



private:
    AABB m_aabb;
    Point3 q;
    Vec3 u, v;
    std::shared_ptr<Material> m_material;
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