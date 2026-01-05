#pragma once

#include "Vec3.h"
#include "Hittable.h"

#include <iostream>

class Quad : public Hittable {
public:
    static std::shared_ptr<Scene> box(const Point3& a, const Point3& b, std::shared_ptr<Material> mat)
    {
        // Returns the 3D box (six sides) that contains the two opposite vertices a & b.
        auto sides = std::make_shared<Scene>();

        // Construct the two opposite vertices with the minimum and maximum coordinates.
        auto min = Point3(std::min(a.x(),b.x()), std::min(a.y(),b.y()), std::min(a.z(),b.z()));
        auto max = Point3(std::max(a.x(),b.x()), std::max(a.y(),b.y()), std::max(a.z(),b.z()));

        auto dx = Vec3(max.x() - min.x(), 0, 0);
        auto dy = Vec3(0, max.y() - min.y(), 0);
        auto dz = Vec3(0, 0, max.z() - min.z());

        sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), max.z()),  dx,  dy, mat)); // front
        sides->add(std::make_shared<Quad>(Point3(max.x(), min.y(), max.z()), -dz,  dy, mat)); // right
        sides->add(std::make_shared<Quad>(Point3(max.x(), min.y(), min.z()), -dx,  dy, mat)); // back
        sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dz,  dy, mat)); // left
        sides->add(std::make_shared<Quad>(Point3(min.x(), max.y(), max.z()),  dx, -dz, mat)); // top
        sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dx,  dz, mat)); // bottom

        return sides;
    }

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