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

    virtual Color emit(const double u, const double v, const Point3& p) const {
        return Color(0.0);
    }
};

class Lambertian : public Material {
public:
    Lambertian(const Color& albedo) : m_albedo(std::make_shared<SolidTexture>(albedo)) {}

    Lambertian(std::shared_ptr<Texture> albedo) : m_albedo(std::move(albedo)) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = hitInfo.normal + Vec3::randomUnitVector();

        if (scatteringDirection.nearZero()) {
            scatteringDirection = hitInfo.normal;
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
        auto scatteringDirection = Vec3::reflect(rayIn.direction(), hitInfo.normal);
        scatteringDirection = scatteringDirection.normalized() + this->fuzz * Vec3::randomUnitVector();
        scatteredRay = Ray(hitInfo.p, scatteringDirection, rayIn.time());
        attenuation = this->albedo;
        return Vec3::dot(hitInfo.normal, scatteringDirection) > 0.0;
    }

private:
    Color albedo;
    double fuzz;
};

class Dielectric : public Material {
    public:
        Dielectric(const double refractiveIndex) : refractiveIndex(refractiveIndex) {}

        bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
            double refractiveIndexReciprocal = hitInfo.front() ? (1.0 / this->refractiveIndex) : this->refractiveIndex;

            auto unitRayInDirection = rayIn.direction().normalized();

            double cosTheta = std::min(Vec3::dot(-unitRayInDirection, hitInfo.normal), 1.0);
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            
            Vec3 scatteringDirection;
            if (refractiveIndexReciprocal * sinTheta > 1.0 || reflectance(cosTheta, refractiveIndexReciprocal) > randomDouble()) {
                scatteringDirection = Vec3::reflect(unitRayInDirection, hitInfo.normal);
            }
            else {
                scatteringDirection = Vec3::refract(unitRayInDirection, hitInfo.normal, refractiveIndexReciprocal);
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

class DiffuseLight : public Material {
public:
    DiffuseLight(std::shared_ptr<Texture> emitter) : m_emitter(std::move(emitter)) {}

    DiffuseLight(const Color& emitColor) : m_emitter(std::make_shared<SolidTexture>(emitColor)) {}

    Color emit(const double u, const double v, const Point3& p) const override {
        return m_emitter->value(u, v, p);
    }

private:
    std::shared_ptr<Texture> m_emitter;
};

class Isotropic : public Material {
  public:
    Isotropic(const Color& albedo) : m_texture(std::make_shared<SolidTexture>(albedo)) {}
    Isotropic(std::shared_ptr<Texture> texture) : m_texture(std::move(texture)) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay)
    const override {
        scatteredRay = Ray(hitInfo.p, Vec3::randomUnitVector(), rayIn.time());
        attenuation = m_texture->value(hitInfo.u, hitInfo.v, hitInfo.p);
        return true;
    }

  private:
    std::shared_ptr<Texture> m_texture;
};