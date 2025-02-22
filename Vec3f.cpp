#include "Vec3f.h"

Vec3f operator+(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() + scalar, rhs.y() + scalar, rhs.z() + scalar);
}

Vec3f operator+(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() + scalar, lhs.y() + scalar, lhs.z() + scalar);
}

Vec3f operator-(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() - scalar, rhs.y() - scalar, rhs.z() - scalar);
}

Vec3f operator-(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() - scalar, lhs.y() - scalar, lhs.z() - scalar);
}

Vec3f operator*(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() * scalar, rhs.y() * scalar, rhs.z() * scalar);
}

Vec3f operator*(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() * scalar, lhs.y() * scalar, lhs.z() * scalar);
}

Vec3f operator/(const float scalar, const Vec3f& rhs) {
    if (scalar == 0.0f)
        return Vec3f(0.0f);
    return Vec3f(rhs.x() / scalar, rhs.y() / scalar, rhs.z() / scalar);
}

Vec3f operator/(const Vec3f& lhs, const float scalar) {
    if (scalar == 0.0f)
        return Vec3f(0.0f);
    return Vec3f(lhs.x() / scalar, lhs.y() / scalar, lhs.z() / scalar);
}