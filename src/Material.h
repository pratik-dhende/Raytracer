#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Hittable.h"
#include "Ray.h"
#include "Texture.h"

class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const {
        return false;
    }
};

class Lambertian : public Material {
public:
    Lambertian(const Color& albedo) : m_albedo(std::make_shared<SolidTexture>(albedo)) {}

    Lambertian(std::shared_ptr<Texture> albedo) : m_albedo(std::move(albedo)) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = hitInfo.getNormal() + Vec3::randomUnitVector();

        if (scatteringDirection.nearZero()) {
            scatteringDirection = hitInfo.getNormal();
        }

        scatteredRay = Ray(hitInfo.p, scatteringDirection, rayIn.time());
        attenuation = m_albedo->value(hitInfo.u, hitInfo.v, hitInfo.p);
        return true;
    }

private:
    std::shared_ptr<Texture> m_albedo;
};

class Metal : public Material {
public:
    Metal(const Color& albedo, const double fuzz) : albedo(albedo), fuzz(std::min(1.0, fuzz)) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = Vec3::reflect(rayIn.direction(), hitInfo.getNormal());
        scatteringDirection = scatteringDirection.normalized() + this->fuzz * Vec3::randomUnitVector();
        scatteredRay = Ray(hitInfo.p, scatteringDirection, rayIn.time());
        attenuation = this->albedo;
        return Vec3::dot(hitInfo.getNormal(), scatteringDirection) > 0.0;
    }

private:
    Color albedo;
    double fuzz;
};

class Dielectric : public Material {
    public:
        Dielectric(const double refractiveIndex) : refractiveIndex(refractiveIndex) {}

        bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
            double refractiveIndexReciprocal = hitInfo.getFront() ? (1.0 / this->refractiveIndex) : this->refractiveIndex;

            auto unitRayInDirection = rayIn.direction().normalized();

            double cosTheta = std::min(Vec3::dot(-unitRayInDirection, hitInfo.getNormal()), 1.0);
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            
            Vec3 scatteringDirection;
            if (refractiveIndexReciprocal * sinTheta > 1.0 || reflectance(cosTheta, refractiveIndexReciprocal) > randomDouble()) {
                scatteringDirection = Vec3::reflect(unitRayInDirection, hitInfo.getNormal());
            }
            else {
                scatteringDirection = Vec3::refract(unitRayInDirection, hitInfo.getNormal(), refractiveIndexReciprocal);
            }

            scatteredRay = Ray(hitInfo.p, scatteringDirection, rayIn.time());
            attenuation = Color(1.0);
            return true;
        }

    private:
        double refractiveIndex;

        static double reflectance(const double cosine, const double refractiveIndexReciprocal) {
            double r0 = (1 - refractiveIndexReciprocal) / (1 + refractiveIndexReciprocal);
            r0 = r0 * r0;
            return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
        }
};