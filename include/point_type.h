//
// Created by hk on 12/8/22.
//

#pragma once
#define PCL_NO_PRECOMPILE

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#ifndef LIO_POINT_TYPE_H
#define LIO_POINT_TYPE_H

struct voxelMap_PointType
{
    PCL_ADD_POINT4D
    union {
        struct {
            float eigenvector_max_x;
            float eigenvector_max_y;
            float eigenvector_max_z;
            float eigenvalue_max;
        };
        float eigen_max[4];
    };

    union {
        struct {
            float eigenvector_mid_x;
            float eigenvector_mid_y;
            float eigenvector_mid_z;
            float eigenvalue_mid;
        };
        float eigen_mid[4];
    };

    union {
        struct {
            float eigenvector_min_x;
            float eigenvector_min_y;
            float eigenvector_min_z;
            float eigenvalue_min;
        };
        float eigen_min[4];
    };

}  EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (voxelMap_PointType,
           (float, x, x) (float, y, y) (float, z, z)
           (float, eigenvector_max_x, eigenvector_max_x) (float, eigenvector_max_y, eigenvector_max_y)
           (float, eigenvector_max_z, eigenvector_max_z) (float, eigenvalue_max, eigenvalue_max)
           (float, eigenvector_mid_x, eigenvector_mid_x) (float, eigenvector_mid_y, eigenvector_mid_y)
           (float, eigenvector_mid_z, eigenvector_mid_z) (float, eigenvalue_mid, eigenvalue_mid)
           (float, eigenvector_min_x, eigenvector_min_x) (float, eigenvector_min_y, eigenvector_min_y)
           (float, eigenvector_min_z, eigenvector_min_z) (float, eigenvalue_min, eigenvalue_min)
)

struct custom_PointType
{
    PCL_ADD_POINT4D
    PCL_ADD_NORMAL4D;
    union {
        struct {
            float intensity;
            float curvature;
            float n_xx; // normal cov element
            float n_xy;
        };
        float data_c[4];
    };

    union {
        struct {
            float n_xz;
            float n_yy;
            float n_yz;
            float n_zz;
        };
        float data_s[4];
    };
    void getNormalCov(Eigen::Matrix3f & m) const
    {
        m(0, 0) = n_xx; m(0, 1) = n_xy; m(0, 2) = n_xz;
        m(1, 0) = n_xy; m(1, 1) = n_yy; m(1, 2) = n_yz;
        m(2, 0) = n_xz; m(2, 1) = n_yz; m(2, 2) = n_zz;
    }
    void recordNormalCov(const Eigen::Matrix3f & m)
    {
        n_xx = m(0, 0); n_xy = m(0, 1); n_xz = m(0, 2);
        n_yy = m(1, 1); n_yz = m(1, 2); n_zz = m(2, 2);
    }
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (custom_PointType,
   (float, x, x) (float, y, y) (float, z, z)
   (float, normal_x, normal_x) (float, normal_y, normal_y) (float, normal_z, normal_z)
   (float, intensity, intensity) (float, curvature, curvature) (float, n_xx, n_xx) (float, n_xy, n_xy)
   (float, n_xz, n_xz) (float, n_yy, n_yy) (float, n_yz, n_yz) (float, n_zz, n_zz)
)

// Velodyne
struct PointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (float, time, time)
)

// Kitti odometry sequence without time
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,
                                   (float, x, x) (float, y, y) (float, z, z)
                                           (float, intensity, intensity)
                                           (uint16_t, ring, ring)
)

// Ouster
struct ousterPointXYZIRT {
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(ousterPointXYZIRT,
      (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
      (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
      (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)


#endif //LIO_POINT_TYPE_H
