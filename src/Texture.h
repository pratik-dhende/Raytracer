#pragma once

#include "Color.h"
#include "Vec3.h"

class Texture {
public:
    virtual ~Texture() {};

    virtual Color value(const double u, const double v, Point3 p) = 0;
};

class SolidTexture : public Texture {
public:
    SolidTexture(const Color& albedo) : m_albedo(albedo) {}

    Color value(const double u, const double v, Point3 p) override {
        return m_albedo;
    }

private:
    Color m_albedo;
};

class CheckerTexture : public Texture {
public:
    CheckerTexture(const double scale, std::shared_ptr<Texture> oddTexture, std::shared_ptr<Texture> evenTexture) : m_inverseScale(1.0 / scale), m_oddTexture(std::move(oddTexture)), m_evenTexture(std::move(evenTexture)) {}

    CheckerTexture(const double scale, const Color& oddColor, const Color& evenColor) : m_inverseScale(1.0 / scale), m_oddTexture(std::make_shared<SolidTexture>(oddColor)), m_evenTexture(std::make_shared<SolidTexture>(evenColor)) {}

    Color value(const double u, const double v, Point3 p) override {
        p *= m_inverseScale;
        Vec3 floored_p = Vec3::floor(p);

        if (static_cast<int>(floored_p.x() + floored_p.y() + floored_p.z()) & 1) {
            return m_oddTexture->value(u, v, p);
        }
        
        return m_evenTexture->value(u, v, p);
    }

private:
    double m_inverseScale;
    std::shared_ptr<Texture> m_oddTexture;
    std::shared_ptr<Texture> m_evenTexture;
};

