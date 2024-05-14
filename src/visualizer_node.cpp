//
// Created by hk on 4/10/24.
//
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
//#include <livox_ros_driver/CustomMsg.h>
#include <ros/package.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "point_type.h"

string root_dir = ROOT_DIR;
//bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

typedef pcl::PointCloud<voxelMap_PointType> voxel_map;
typedef pcl::PointCloud<voxelMap_PointType>::Ptr voxel_map_ptr;

float max_color_scale = 0.0;
float use_alpha = 0.8;
float max_color_coef;

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin) {
        v = vmin;
    }

    if (v > vmax) {
        v = vmax;
    }

    double dr, dg, db;

    if (v < 0.1242) {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    } else if (v < 0.3747) {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    } else if (v < 0.6253) {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    } else if (v < 0.8758) {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    } else {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t)(255 * dr);
    g = (uint8_t)(255 * dg);
    b = (uint8_t)(255 * db);
}

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q) {

    Eigen::Matrix3d rot;
    rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
            z_vec(1), z_vec(2);
    Eigen::Matrix3d rotation = rot.transpose();
    Eigen::Quaterniond eq(rotation);
    q.w = eq.w();
    q.x = eq.x();
    q.y = eq.y();
    q.z = eq.z();
}

int plane_id = 0;
void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string plane_ns, const voxelMap_PointType & plane_data,
                    const float alpha, const Eigen::Vector3d rgb) {
    visualization_msgs::Marker plane;
    plane.header.frame_id = "world";
    plane.header.stamp = ros::Time();
    plane.ns = plane_ns;
    plane.id = plane_id++;
    plane.type = visualization_msgs::Marker::CYLINDER;
    plane.action = visualization_msgs::Marker::ADD;
    plane.pose.position.x = plane_data.x;
    plane.pose.position.y = plane_data.y;
    plane.pose.position.z = plane_data.z;
    V3D eigenvector_max(plane_data.eigenvector_max_x, plane_data.eigenvector_max_y, plane_data.eigenvector_max_z);
    V3D eigenvector_mid(plane_data.eigenvector_mid_x, plane_data.eigenvector_mid_y, plane_data.eigenvector_mid_z);
    V3D eigenvector_min(plane_data.eigenvector_min_x, plane_data.eigenvector_min_y, plane_data.eigenvector_min_z);
    geometry_msgs::Quaternion q;
    CalcVectQuation(eigenvector_max, eigenvector_mid, eigenvector_min, q);
    plane.pose.orientation = q;
    plane.scale.x = 3 * sqrt(plane_data.eigenvalue_max);
    plane.scale.y = 3 * sqrt(plane_data.eigenvalue_mid);
    plane.scale.z = 2 * sqrt(plane_data.eigenvalue_min);
    plane.color.a = alpha;
    plane.color.r = rgb(0);
    plane.color.g = rgb(1);
    plane.color.b = rgb(2);
    plane.lifetime = ros::Duration();
    plane_pub.markers.push_back(plane);
}

void pubVoxelMap(const voxel_map_ptr & vm_ptr,
//                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {


    visualization_msgs::MarkerArray voxel_plane;
    voxel_plane.markers.reserve(1000000);
    for (size_t i = 0; i < vm_ptr->size(); i++) {
        const voxelMap_PointType & voxel_data = vm_ptr->points[i];
//        double trace = voxel_data.eigenvalue_max + voxel_data.eigenvalue_mid + voxel_data.eigenvalue_min;
        float color_scale = voxel_data.eigenvalue_min;

        if (color_scale >= max_color_scale) {
            color_scale = max_color_scale;
        }
        color_scale = color_scale * (1.0 / max_color_scale);
//        trace = pow(trace, pow_num);

        uint8_t r, g, b;
        mapJet(color_scale, 0, 1, r, g, b);
        Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
        double alpha;
        alpha = use_alpha;
//        if (pub_plane_list[i].is_plane) {
//            alpha = use_alpha;
//        } else {
//            alpha = 0;
//        }
        pubSinglePlane(voxel_plane, "plane", voxel_data, alpha, plane_rgb);
    }
    plane_map_pub.publish(voxel_plane);
//    loop.sleep();
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

//    // visualization params
    nh.param<float>("max_color_scale", max_color_scale, 0.5);
//    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
//
//    string pose_target_file = root_dir + "/Log/target_path.txt";
//    string pos_target_end_time = root_dir + "/Log/target_end_path.txt";

    ros::Publisher voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    string voxel_map_file = root_dir + "/Log/voxel_map.pcd";
    voxel_map_ptr planes(new voxel_map);
    pcl::io::loadPCDFile(voxel_map_file, *planes);
    cout << *planes << endl;


    double pow_num = 0.2;
    ros::Rate loop(500);

//    for (size_t i = 0; i < planes->size(); i++) {
//        const voxelMap_PointType &voxel_data = planes->points[i];
//        float color_scale = voxel_data.eigenvalue_min;
//        max_color_scale = max(color_scale, max_color_scale);
//    }
//    max_color_scale *= max_color_coef;
//    cout << "max_color_scale" << max_color_scale << endl;
    ROS_INFO("max_color_scale: %f", max_color_scale);

//    signal(SIGINT, SigHandle);
    bool is_published = false;
    ros::Rate rate(1000);
//    bool status = ros::ok();
    while ( ros::ok())
    {
        /******* Publish voxel map *******/
        if (!is_published && voxel_map_pub.getNumSubscribers() > 0) {
            pubVoxelMap(planes, voxel_map_pub);
            is_published = true;
        }
//        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
