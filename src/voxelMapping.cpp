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

#include "preprocess.h"
#include "voxel_map_util.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false;
bool   time_sync_en = false, extrinsic_est_en = true, path_en = true;
double lidar_time_offset = 0.0;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string  lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;


vector<vector<int>>  pointSearchInd_surf;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
//std::vector<M3D> var_down_body;
std::vector<M3D> cov_measurement;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

// parameters for normal estimation
int N_SCAN, Horizon_SCAN;
string ring_table_dir;
bool check_normal;
double incident_cov_scale, roughness_cov_scale, trace_scale;
double incident_cov_max, roughness_cov_max;
double visual_ray_scale, visual_tan_scale, visual_a_scale;
double local_tan_scale, local_rad_scale;
float roughness_max;

double normal_cov_threshold, lambda_cov_threshold;
int normal_cov_update_interval, normal_cov_incre_min;
int num_update_thread;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
int num_voxel_full = 0;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

//for ground truth
geometry_msgs::PoseStamped msg_target_pose;
nav_msgs::Path path_target_begin, path_target_end;
// for ground truth, target in IMU frame
vector<double>       gt_extrinT(3, 0.0);
vector<double>       gt_extrinR(9, 0.0);
V3D gt_T_wrt_IMU(Zero3d);
M3D gt_R_wrt_IMU(Eye3d);

// record begin time for pose interpolation
deque<double> timestamps_lidar;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

string bag_file_input;
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
//    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

double mean_preprocess = 0.0;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
//    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    timestamps_lidar.push_back(last_timestamp_lidar);
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mean_preprocess = mean_preprocess * (scan_count - 1) / scan_count + s_plot11[scan_count] / scan_count;
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    if (runtime_pos_log)
        printf("[ pre-process ]: this time: %0.6f ms, mean : %0.6f ms\n", s_plot11[scan_count] * 1000, mean_preprocess * 1000);
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
//void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
//{
//    mtx_buffer.lock();
//    double preprocess_start_time = omp_get_wtime();
//    scan_count ++;
//    if (msg->header.stamp.toSec() < last_timestamp_lidar)
//    {
//        ROS_ERROR("lidar loop back, clear buffer");
//        lidar_buffer.clear();
//    }
//    last_timestamp_lidar = msg->header.stamp.toSec();
//
//    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
//    {
//        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
//    }
//
//    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
//    {
//        timediff_set_flg = true;
//        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
//        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
//    }
//
//    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
//    p_pre->process(msg, ptr);
//    lidar_buffer.push_back(ptr);
//    time_buffer.push_back(last_timestamp_lidar);
//
//    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
//    mtx_buffer.unlock();
//    sig_buffer.notify_all();
//}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
//        ROS_WARN("imu loop back, clear buffer");
//        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10
        || std::abs(msg->angular_velocity.y) > 10
        || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z
        );
        return;
    }

//    // 如果是第一帧 拿过来做重力对齐
//    // TODO 用多帧平均的重力
//    if (is_first_imu) {
//        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
//
//        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
//
//        is_first_imu = false;
//    }

    last_timestamp_imu = timestamp;

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {

//            std::printf("\nFirst 100 points: \n");
//            for(int i=0; i < 100; ++i){
//                std::printf("%f ", meas.lidar->points[i].curvature  / double(1000));
//            }
//
//            std::printf("\n Last 100 points: \n");
//            for(int i=100; i >0; --i){
//                std::printf("%f ", meas.lidar->points[meas.lidar->size() - i - 1].curvature / double(1000));
//            }
//            std::printf("last point offset time: %f\n", meas.lidar->points.back().curvature / double(1000));
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
//            lidar_end_time = meas.lidar_beg_time + (meas.lidar->points[meas.lidar->points.size() - 20]).curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
//            std::printf("pcl_bag_time: %f\n", meas.lidar_beg_time);
//            std::printf("lidar_end_time: %f\n", lidar_end_time);
        }

        meas.lidar_end_time = lidar_end_time;
//        std::printf("Scan start timestamp: %f, Scan end time: %f\n", meas.lidar_beg_time, meas.lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}


PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());


void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size; i++)
        {
            PointType const * const p = &laserCloudFullRes->points[i];
//            if(p->intensity < 5){
//                continue;
//            }
//            if (p->x < 0 and p->x > -4
//                    and p->y < 1.5 and p->y > -1.5
//                            and p->z < 2 and p->z > -1) {
//                continue;
//            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
//            if (p_world.intensity < 0.01)
//                p_world.intensity = 0.01;
//            if (p_world.z > 1) {
//                continue;
//            }
            laserCloudWorld.push_back(p_world);
//            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
//    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));

    tf::Transform                   transform_body_lidar;
    tf::Quaternion                  q_body_lidar;
    transform_body_lidar.setOrigin(tf::Vector3(state_point.offset_T_L_I(0), \
                                    state_point.offset_T_L_I(1), \
                                    state_point.offset_T_L_I(2)));
    Eigen::Quaterniond eq_body_lidar(state_point.offset_R_L_I);
    q_body_lidar.setW(eq_body_lidar.w());
    q_body_lidar.setX(eq_body_lidar.x());
    q_body_lidar.setY(eq_body_lidar.y());
    q_body_lidar.setZ(eq_body_lidar.z());
    transform_body_lidar.setRotation( q_body_lidar );
    br.sendTransform( tf::StampedTransform( transform_body_lidar, odomAftMapped.header.stamp, "body", "lidar" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 1 == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }

    {
        vect3 pos_target;
        pos_target = state_point.pos + state_point.rot * gt_T_wrt_IMU;
        Eigen::Quaterniond quat_target(state_point.rot * gt_R_wrt_IMU);
//        Eigen::Quaterniond quat_target(T.block<3, 3>(0, 0));

        msg_target_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
        msg_target_pose.header.frame_id = "camera_init";
        msg_target_pose.pose.position.x = pos_target(0);
        msg_target_pose.pose.position.y = pos_target(1);
        msg_target_pose.pose.position.z = pos_target(2);
        msg_target_pose.pose.orientation.x = quat_target.x();
        msg_target_pose.pose.orientation.y = quat_target.y();
        msg_target_pose.pose.orientation.z = quat_target.z();
        msg_target_pose.pose.orientation.w = quat_target.w();
        path_target_end.poses.push_back(msg_target_pose);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        PointType p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

void transformCloudNormalOMP(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    int point_size = input_cloud->size();
    trans_cloud->resize(point_size);

    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < point_size; i++) {
        const PointType & p_c = input_cloud->points[i];
        PointType & p_out = trans_cloud->points[i];
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_c.getVector3fMap().cast<double>() + state_point.offset_T_L_I) + state_point.pos;
        p_out.x = p_body(0);
        p_out.y = p_body(1);
        p_out.z = p_body(2);
        p_out.intensity = p_c.intensity;

        // normal
        V3F normal_global(state_point.rot.cast<float>() * state_point.offset_R_L_I.cast<float>() * p_c.getNormalVector3fMap());
        normal_global.normalize();
        p_out.normal_x = normal_global(0);
        p_out.normal_y = normal_global(1);
        p_out.normal_z = normal_global(2);
    }
}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    // cov: body <--lidar
    M3D COV_body =
            state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate()
            + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate()
            + il_t_var;

    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate()
        + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
        + t_var;

    return COV_world;
}

void prepareMapping(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud,
                    vector<pointWithCov>& pv_list)
{
    int point_size = input_cloud->size();
    pv_list.resize(point_size);

    V3D lid_pos = state_point.rot * state_point.offset_T_L_I + state_point.pos;
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < point_size; i++) {
        const PointType & p_c = input_cloud->points[i];
        pointWithCov & pv = pv_list[i];
        // can be simplified
        pv.point = state_point.rot * (state_point.offset_R_L_I * p_c.getVector3fMap().cast<double>() + state_point.offset_T_L_I) + state_point.pos;

        // normal
        V3F normal_global(state_point.rot.cast<float>() * state_point.offset_R_L_I.cast<float>() * p_c.getNormalVector3fMap());
        pv.normal = normal_global.normalized();

//        pv.cov_lidar = cov_measurement[i]; // cov in lidar frame
        pv.cov = transformLiDARCovToWorld(pv.point, kf, cov_measurement[i]); // cov in world frame

        pv.ray = pv.point - lid_pos;
        pv.p2lidar = pv.ray.norm();
        pv.ray.normalize();
        pv.tangent_cov = pow(DEG2RAD(angle_cov), 2) * pv.p2lidar * pv.p2lidar;
    }
}

void transformCloudNormalCopyCovOMP(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud,
                                vector<pointWithCov>& pv_list)
{
    int point_size = input_cloud->size();
    pv_list.resize(point_size);

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < point_size; i++) {
        const PointType & p_c = input_cloud->points[i];
        pointWithCov & pv = pv_list[i];
        pv.point_world = state_point.rot * (state_point.offset_R_L_I * p_c.getVector3fMap().cast<double>() + state_point.offset_T_L_I) + state_point.pos;

        // normal
        V3F normal_global(state_point.rot.cast<float>() * state_point.offset_R_L_I.cast<float>() * p_c.getNormalVector3fMap());
        pv.normal = normal_global.normalized();

        pv.point << p_c.x, p_c.y, p_c.z;
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        pv.cov_lidar = cov_measurement[i]; // cov in lidar frame
        pv.cov = transformLiDARCovToWorld(pv.point, kf, cov_measurement[i]); // cov in world frame
    }
}

double roug_cos_angle = cos(20.0 / 180.0 * M_PI);
void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
//    laserCloudOri->clear();
//    corr_normvect->clear();
    feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);

    transformCloudNormalCopyCovOMP(s, feats_down_body, pv_list);
    V3F body_pos(s.pos.x(), s.pos.y(), s.pos.z());

    // ===============================================================================================================
    double match_start = omp_get_wtime();
    std::vector<ptpl> ptpl_list;
    std::vector<V3D> non_match_list;
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, body_pos,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1){
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1);

    V3D lid_pos = s.rot * s.offset_T_L_I + s.pos;
    SO3 r_lid_wrt_w = s.rot * s.offset_R_L_I; // rotation lidar wrt. world
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
//        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);//lidar frame
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; //body frame
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this); //body frame: p^

        /*** get the normal vector of closest surface/corner ***/
//        const PointType &norm_p = corr_normvect->points[i];
//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
//        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x()
                + norm_vec.y() * ptpl_list[i].point_world.y()
                + norm_vec.z() * ptpl_list[i].point_world.z()
                + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        /*** Covariance ***/
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar; // point cov in lidar frame

        // consider incident and roughness
//        if (scan_num > 20)
        {
            double cos_theta = ptpl_list[i].normal.dot(ptpl_list[i].point_normal.cast<double>()); // [-1.0, 1.0]
            double roughness = roughness_cov_scale;
            if (cos_theta > 0)
                roughness = ( 1 - cos_theta * cos_theta) * roughness_cov_scale; // sin^2_theta * roughness
            V3D ray_world = ptpl_list[i].point_world - lid_pos;
            double dist = ray_world.norm();
            ray_world.normalize();
            double incident_scale = calcIncidentCovScale(ray_world, dist, ptpl_list[i].normal);
            V3D ray_lidar = ptpl_list[i].point;
            ray_lidar.normalize();
            cov_lidar = cov_lidar + roughness * M3D::Identity() +
                        incident_scale * ray_lidar * ray_lidar.transpose();
        }
//        // cov in the world should consider all the factors including lidar, IMU, pose, extrinsic timestamp ....
//        M3D R_cov_Rt = transformLiDARCovToWorld(ptpl_list[i].point, kf, cov_lidar); // cov in world frame
        M3D R_cov_Rt = r_lid_wrt_w * cov_lidar * r_lid_wrt_w.conjugate();
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }

    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
    // std::printf("res_mean: %f\n", res_mean_last);
    // std::printf("ef_num: %d\n", effct_feat_num);
}

void initCloudCovOMP()
{
//    var_down_body.resize(feats_down_size);
    cov_measurement.resize(feats_down_size);
    #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; ++i) {
        const PointType & pt = feats_down_body->points[i];
        V3D point_this(pt.x, pt.y, pt.z);
//                var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
//        cov_measurement[i] = calcBodyCov(point_this, ranging_cov, angle_cov)
//                             + calcRoughCov(pt.intensity, point_this); // intensity record roughness
        cov_measurement[i] = calcBodyCov(point_this, ranging_cov, angle_cov);
//        var_down_body[i] = cov_measurement[i];
    }
}

void saveTraj(const string & pose_target_file)
{
    //also interpolate pose at lidar begin time
    int begin_time_ptr = 0;
    int begin_time_size = timestamps_lidar.size();
    double begin_time = timestamps_lidar[0];
//        int end_time_ptr_left = 0;
    double end_time_left = path_target_end.poses[0].header.stamp.toSec();
    while (begin_time_ptr < begin_time_size) {
        begin_time = timestamps_lidar[begin_time_ptr];
        if (begin_time > end_time_left)
            break;
        ++begin_time_ptr;
    }

    printf("\n..............Saving path................\n");
    printf("path file: %s\n", pose_target_file.c_str());
    ofstream of_beg(pose_target_file);
    of_beg.setf(ios::fixed, ios::floatfield);
    of_beg.precision(12);
    of_beg<< path_target_end.poses[0].header.stamp.toSec()<< " "
          <<path_target_end.poses[0].pose.position.x<< " "
          <<path_target_end.poses[0].pose.position.y<< " "
          <<path_target_end.poses[0].pose.position.z<< " "
          <<path_target_end.poses[0].pose.orientation.x<< " "
          <<path_target_end.poses[0].pose.orientation.y<< " "
          <<path_target_end.poses[0].pose.orientation.z<< " "
          <<path_target_end.poses[0].pose.orientation.w<< "\n";

    for (int i = 1; i < path_target_end.poses.size(); ++i) {
        double end_time_right = path_target_end.poses[i].header.stamp.toSec();
//                printf("end time left: %f\n", end_time_left);
        while (begin_time_ptr < begin_time_size && timestamps_lidar[begin_time_ptr] < end_time_right) {
            begin_time = timestamps_lidar[begin_time_ptr];
            if (abs(begin_time - end_time_right) < 0.00001 || abs(begin_time - end_time_left) < 0.00001) {
                ++begin_time_ptr;
                continue;
            }
            //interpolate between end time left and right
            double dt_l = begin_time - end_time_left;
            double dt_r = end_time_right - begin_time;
            double dt_l_r = end_time_right - end_time_left;
            double ratio_l = dt_l / dt_l_r;
            double ratio_r = dt_r / dt_l_r;

            const auto &pose_l = path_target_end.poses[i - 1].pose;
            const auto &pose_r = path_target_end.poses[i].pose;

            V3D pos_l(pose_l.position.x, pose_l.position.y, pose_l.position.z);
            V3D pos_r(pose_r.position.x, pose_r.position.y, pose_r.position.z);

            Eigen::Quaterniond q_l(pose_l.orientation.w, pose_l.orientation.x, pose_l.orientation.y,
                                   pose_l.orientation.z);
            Eigen::Quaterniond q_r(pose_r.orientation.w, pose_r.orientation.x, pose_r.orientation.y,
                                   pose_r.orientation.z);

            Eigen::Quaterniond  q_begin_time = q_l.slerp(ratio_l, q_r);
            V3D pos_begin_time = pos_l * ratio_r + pos_r * ratio_l;

            of_beg<< begin_time << " "
                  <<pos_begin_time(0)<< " " <<pos_begin_time(1)<< " " <<pos_begin_time(2)<< " "
                  <<q_begin_time.x()<< " "
                  <<q_begin_time.y()<< " "
                  <<q_begin_time.z()<< " "
                  <<q_begin_time.w()<< "\n";

            ++begin_time_ptr;
        }
//                if (abs(begin_time - end_time_right) < 0.000001)
//                    ++begin_time_ptr;
//                printf("end_time_right: %f\n", end_time_right);

        of_beg<< path_target_end.poses[i].header.stamp.toSec()<< " "
              <<path_target_end.poses[i].pose.position.x<< " "
              <<path_target_end.poses[i].pose.position.y<< " "
              <<path_target_end.poses[i].pose.position.z<< " "
              <<path_target_end.poses[i].pose.orientation.x<< " "
              <<path_target_end.poses[i].pose.orientation.y<< " "
              <<path_target_end.poses[i].pose.orientation.z<< " "
              <<path_target_end.poses[i].pose.orientation.w<< "\n";
        end_time_left = end_time_right;
    }
    of_beg.close();
}

void saveVoxelMap(const string& voxel_map_file)
{
    printf("\n..........Saving Voxel Map...........\n");
    printf("voxel map file: %s\n", voxel_map_file.c_str());
    std::vector<Plane> pub_plane_list;
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        GetUpdatePlane(iter->second, publish_max_voxel_layer, pub_plane_list);
    }
    pcl::PointCloud<voxelMap_PointType> planes;
    planes.resize(pub_plane_list.size());
    for (size_t i = 0; i < pub_plane_list.size(); i++) {
        const Plane & pl = pub_plane_list[i];
        voxelMap_PointType & pl_output = planes.points[i];
        pl_output.x = pl.center[0];
        pl_output.y = pl.center[1];
        pl_output.z = pl.center[2];

        pl_output.eigenvalue_max = pl.max_eigen_value;
        pl_output.eigenvalue_mid = pl.mid_eigen_value;
        pl_output.eigenvalue_min = pl.min_eigen_value;

        pl_output.eigenvector_max_x = pl.x_normal[0];
        pl_output.eigenvector_max_y = pl.x_normal[1];
        pl_output.eigenvector_max_z = pl.x_normal[2];

        pl_output.eigenvector_mid_x = pl.y_normal[0];
        pl_output.eigenvector_mid_y = pl.y_normal[1];
        pl_output.eigenvector_mid_z = pl.y_normal[2];

        pl_output.eigenvector_min_x = pl.normal[0];
        pl_output.eigenvector_min_y = pl.normal[1];
        pl_output.eigenvector_min_z = pl.normal[2];
    }
    pcl::io::savePCDFile(voxel_map_file, planes);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/lidar_time_offset", lidar_time_offset, 0.0);

    // mapping algorithm params
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
    nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
    nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/lidar_type", p_imu->lidar_type, AVIA);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/Horizon_SCAN", p_pre->Horizon_SCAN, 1800);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    // Log
    nh.param<bool>("log/enable", runtime_pos_log, false);

    // cov scale
    nh.param<double>("cov_scale/roughness_cov_scale", roughness_cov_scale, 1.0);
    nh.param<double>("cov_scale/trace_scale", trace_scale, 1.0);
    nh.param<double>("cov_scale/local_tan_scale", local_tan_scale, 1.0);
    nh.param<double>("cov_scale/local_rad_scale", local_rad_scale, 1.0);
    nh.param<double>("cov_scale/incident_cov_scale", incident_cov_scale, 1.0);
    nh.param<double>("cov_scale/incident_cov_max", incident_cov_max, 0.1);
    nh.param<double>("cov_scale/roughness_cov_max", roughness_cov_max, 0.1);
    nh.param<double>("cov_scale/visual_ray_scale", visual_ray_scale, 111.1);
    nh.param<double>("cov_scale/visual_tan_scale", visual_tan_scale, 111.1);
    nh.param<double>("cov_scale/visual_a_scale", visual_a_scale, 0.5);

    // cov incremental
    nh.param<double>("cov_incremental/normal_cov_threshold", normal_cov_threshold, 0.0001);
    nh.param<double>("cov_incremental/lambda_cov_threshold", lambda_cov_threshold, 0.0001);
    nh.param<int>("cov_incremental/normal_cov_update_interval", normal_cov_update_interval, 50);
    nh.param<int>("cov_incremental/normal_cov_incre_min", normal_cov_incre_min, 50);
    nh.param<int>("cov_incremental/num_update_thread", num_update_thread, 4);

    // ring Fals Normal Estimation parameters
    std::string PROJECT_NAME = "lio";
    nh.param<string>("normal/project_name", PROJECT_NAME, "/tmp");
    nh.param<bool>("normal/compute_table", p_pre->compute_table, false);
    nh.param<bool>("normal/compute_normal", p_pre->compute_normal, false);
    nh.param<bool>("normal/check_normal", check_normal, true);
    nh.param<float>("normal/roughness_max", p_pre->roughness_max, 2.0);
    nh.param<float>("normal/roughness_max", roughness_max, 2.0);
    nh.param<string>("normal/ring_table_dir", ring_table_dir, "/tmp");
    std::string pkg_path = ros::package::getPath(PROJECT_NAME);
    ring_table_dir = pkg_path + ring_table_dir;
    p_pre->ring_table_dir = ring_table_dir;
    p_pre->runtime_log = runtime_pos_log;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    p_pre->initNormalEstimator();

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // XXX 暂时现在lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));


    // for ground truth target
    nh.param<vector<double>>("ground_truth/extrinsic_T", gt_extrinT, vector<double>());
    nh.param<vector<double>>("ground_truth/extrinsic_R", gt_extrinR, vector<double>());
    gt_T_wrt_IMU<<VEC_FROM_ARRAY(gt_extrinT);
    gt_R_wrt_IMU<<MAT_FROM_ARRAY(gt_extrinR);
    FILE *fp_target;

    bag_file_input = "target_path.txt";
    boost::filesystem::path bag_file(bag_file_input);
    string pose_target_file = root_dir + "/Log/target_path.txt";
    string pos_target_end_time = root_dir + "/Log/target_end_path.txt";

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
//    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
//        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
//        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/Odometry", 100000);
    ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>
            ("/Extrinsic", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 100000);
    ros::Publisher voxel_map_pub =
            nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    ros::Publisher map_ringfals_pub = nh.advertise<visualization_msgs::Marker>("/map_ring_fals", 10);
    ros::Publisher rough_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/rough_cov", 10);
    ros::Publisher map_voxel_normal_pub = nh.advertise<visualization_msgs::Marker>("/map_voxel_normal", 10);
    ros::Publisher scan_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/scan_cov", 10);
    ros::Publisher scan_cov_ir_pub = nh.advertise<visualization_msgs::MarkerArray>("/scan_cov_ir", 10);
//------------------------------------------------------------------------------------------------------
    // for Plane Map
    bool init_map = false;

    double sum_optimize_time = 0, sum_update_time = 0;
    double mean_scan_time = 0;
    int scan_index = 0;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();

        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0 = omp_get_wtime();
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            // ===============================================================================================================
            if (flg_EKF_inited && !init_map) {
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                feats_down_body = feats_undistort;
                feats_down_size = feats_down_body->points.size();
                std::vector<pointWithCov> pv_list(feats_down_size);

                std::cout << kf.get_P() << std::endl;

                initCloudCovOMP();
                prepareMapping(state_point, feats_down_body, pv_list);
                buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size, max_points_size, min_eigen_value,
                              voxel_map);
                std::cout << "build voxel map" << std::endl;

                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                    publish_frame_world(pubLaserCloudFull);
                    publish_frame_body(pubLaserCloudFull_body);
                }
                init_map = true;
                continue;
            }

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            double t_ds = omp_get_wtime();
            downSizeFilterSurf.filter(*feats_down_body);
//            ROS_WARN("downSizeFilterSurf cost: %.3fs", ( omp_get_wtime() - t_ds) * 1000.0);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            // todo custom downsample for roughness and normal,  normalize in S2 manifold
            for (PointType p : feats_down_body->points)
                p.getNormalVector3fMap().normalize();

            feats_down_size = feats_down_body->points.size();
            initCloudCovOMP();

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            // ===============================================================================================================
            // 开始迭代滤波
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            // todo
            kf.update_iterated_dyn_share_diagonal();
//            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            double t_update_end = omp_get_wtime();
            sum_optimize_time += t_update_end - t_update_start;

            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
//
            std::printf("BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
                        kf.get_x().ba.x(),kf.get_x().ba.y(),kf.get_x().ba.z(),
                        kf.get_x().bg.x(),kf.get_x().bg.y(),kf.get_x().bg.z(),
                        kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z()
            );

            // ===============================================================================================================
            /*** add the points to the voxel map ***/
            std::vector<pointWithCov> pv_list(feats_down_size);
            prepareMapping(state_point, feats_down_body, pv_list);

            t_update_start = omp_get_wtime();
            std::sort(pv_list.begin(), pv_list.end(), var_contrast);
//            updateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
//                              max_points_size, max_points_size, min_eigen_value,
//                              voxel_map);
            updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                           max_points_size, max_points_size, min_eigen_value,
                           voxel_map);
            t_update_end = omp_get_wtime();
            sum_update_time += t_update_end - t_update_start;

            double t1 = omp_get_wtime();
            mean_scan_time += t1 - t0;
            scan_index++;
            std::printf("Mean Time: opt %.3fms  update: %.3fms\n", sum_optimize_time / scan_index * 1000.0, sum_update_time / scan_index * 1000.0);
            std::printf("Mean time: scan %.3fms.\n", mean_scan_time / scan_index * 1000.0);

            // ===============================================================================================================
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (publish_voxel_map && voxel_map_pub.getNumSubscribers() > 0) {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            if (map_ringfals_pub.getNumSubscribers() > 0)
                pubMapRingFalsNormal(voxel_map, publish_max_voxel_layer, ros::Time().fromSec(lidar_end_time),
                                     map_ringfals_pub);
            if (map_voxel_normal_pub.getNumSubscribers() > 0)
                pubMapVoxelNormal(voxel_map, publish_max_voxel_layer, ros::Time().fromSec(lidar_end_time),
                                  map_voxel_normal_pub);
            if (scan_cov_pub.getNumSubscribers() > 0)
                pubScanWithCov(scan_cov_pub, pv_list, "world");
            if (scan_cov_ir_pub.getNumSubscribers() > 0)
                pubScanWithCov(scan_cov_ir_pub, pv_list, "world", true);
            if (rough_cov_pub.getNumSubscribers() > 0)
                pubScanRoughness(rough_cov_pub, feats_undistort);
        }

        status = ros::ok();
        rate.sleep();
    }

    if (p_pre->compute_table) {
        printf(".....Computing M inverse....\n");
        p_pre->range_image.computeMInverse();
        printf("Computing M inverse matrix.\n");
        printf(".....Saving range image lookup table....\n");
        p_pre->range_image.saveLookupTable(p_pre->ring_table_dir, "ring" + std::to_string(N_SCAN));
    }

    //save globalPath
    saveTraj(pose_target_file);

    //save final voxel map
    string voxel_map_file = root_dir + "/Log/voxel_map.pcd";
    saveVoxelMap(voxel_map_file);

    return 0;
}
