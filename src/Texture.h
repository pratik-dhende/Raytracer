#pragma once

#include "Color.h"
#include "Vec3.h"
#include "rtw_stb_image.h"
#include "PerlinNoise.h"

class Texture {
public:
    virtual ~Texture() {};

    virtual Color value(double u, double v, const Point3& p) const = 0;
};

class SolidTexture : public Texture {
public:
    SolidTexture(const Color& albedo) : m_albedo(albedo) {}

    Color value(const double u, const double v, const Point3& p) const override {
        return m_albedo;
    }

private:
    Color m_albedo;
};

class CheckerTexture : public Texture {
public:
    CheckerTexture(const double scale, std::shared_ptr<Texture> oddTexture, std::shared_ptr<Texture> evenTexture) : m_inverseScale(1.0 / scale), m_oddTexture(std::move(oddTexture)), m_evenTexture(std::move(evenTexture)) {}

    CheckerTexture(const double scale, const Color& oddColor, const Color& evenColor) : m_inverseScale(1.0 / scale), m_oddTexture(std::make_shared<SolidTexture>(oddColor)), m_evenTexture(std::make_shared<SolidTexture>(evenColor)) {}

    Color value(double u, double v, const Point3& p) const override {
        const Vec3 floored_p = Vec3::floor(p * m_inverseScale);

        if ((static_cast<int>(floored_p.x()) + static_cast<int>(floored_p.y()) + static_cast<int>(floored_p.z())) & 1) {
            return m_oddTexture->value(u, v, p);
        }
        
        return m_evenTexture->value(u, v, p);
    }

private:
    double m_inverseScale;
    std::shared_ptr<Texture> m_oddTexture;
    std::shared_ptr<Texture> m_evenTexture;
};

class ImageTexture : public Texture {
public:
    ImageTexture(const char* filename) : m_image(filename) {}

    Color value(double u, double v, const Point3& p) const override {
        if (m_image.height() <= 0) {
            return Color(0.0, 1.0, 1.0);
        }

        int i = static_cast<int>(std::clamp(u, 0.0, 1.0) * m_image.width());
        int j = static_cast<int>(std::clamp(1.0 - v, 0.0, 1.0) * m_image.height());

        auto pixel = m_image.pixel_data(i, j);

        return Color(pixel[0], pixel[1], pixel[2]) * (1.0 / 255.0);
    }

private:
    rtw_image m_image;
};

class PerlinNoiseTexture : public Texture{
public:
    PerlinNoiseTexture(const double frequency) : m_frequency(frequency) {}

    Color value(double u, double v, const Point3& p) const override {
        return Color(.5, .5, .5) * (1 + std::sin(m_frequency * p.z() + 10 * m_noise.turbulence(p, 7)));
    }

private:
    PerlinNoise m_noise;
    double m_frequency;
};

