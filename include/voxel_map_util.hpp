#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
#include "common_lib.h"
#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
//#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define HASH_P 116101
#define MAX_N 10000000000

static int plane_id = 0;
extern int refine_maximum_iter, refine_en;
extern bool check_normal;
extern double incident_cov_scale, roughness_cov_scale, trace_scale, trace_outlier_t;
extern double local_tan_scale, local_rad_scale, angle_cov, ranging_cov;
extern double roughness_cov_max, incident_cov_max;
extern double visual_ray_scale, visual_tan_scale, visual_a_scale;
double incident_cos_min = cos(75.0 / 180.0 * M_PI);
double angle_threshold = 0.1; // degree
double dist_threshold = 0.001;

extern double normal_cov_threshold, lambda_cov_threshold;
extern int normal_cov_update_interval, normal_cov_incre_min, num_update_thread;

// a point to plane matching structure
typedef struct ptpl {
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Vector3d normal;
  Eigen::Vector3f point_normal;
  Eigen::Vector3d center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
  Eigen::Matrix3d cov_lidar;
} ptpl;

// ray: unit vector
double calcIncidentCovScale(const V3D & ray, const double & dist, const V3D& normal)
{
    static double angle_rad = DEG2RAD(angle_cov);
    double cos_incident = abs(normal.dot(ray));
    cos_incident = max(cos_incident, incident_cos_min);
    double sin_incident = sqrt(1 - cos_incident * cos_incident);
    double sigma_a = dist * sin_incident / cos_incident * angle_rad; // range * tan(incident) * sigma_angle
    return min(incident_cov_max, incident_cov_scale * sigma_a * sigma_a); // scale * sigma_a^2
}

// dir must be a unit vector
void findLocalTangentBases(const V3D& dir, V3D & base1, V3D & base2)
{
    // find base vector in the local tangent space
    base1 = dir.cross(V3D(1.0, 0, 0));
    if (dir(0) == 1.0)
        base1 = dir.cross(V3D(0, 0, 1.0));
    base1.normalize();
    base2 = dir.cross(base1);
    base2.normalize();
}

// 3D point with covariance
typedef struct pointWithCov {
  Eigen::Vector3d point; // in lidar frame
  Eigen::Vector3d point_world;
  Eigen::Matrix3d cov; // world frame
  Eigen::Matrix3d cov_lidar;
  Eigen::Vector3f normal;
  Eigen::Vector3d ray; // point to lidar(t), a unit vector
  double p2lidar;
//  double roughness_cov; // ^2, isotropic cov in 3D space
  double tangent_cov; // ^2, isotropic cov in tangent space

  void getCovValues(double & range_cov, double & tan_cov, const V3D & n) const
  {
      double incident_scale = calcIncidentCovScale(ray, p2lidar, n);
//      range_cov = ranging_cov * ranging_cov + roughness_cov + incident_scale;
      range_cov = ranging_cov * ranging_cov + incident_scale;
      tan_cov = tangent_cov;
  }

  M3D getCovInv() const
  {
      double range_cov, tan_cov;
      getCovValues(range_cov, tan_cov, normal.cast<double>());
      double range_var_inv = 1.0 / sqrt(range_cov); // 1 / sigma range
      double tangent_var_inv = 1.0 / sqrt(tan_cov); // 1 / (range * sigma angle)

      V3D b1, b2;
      findLocalTangentBases(ray, b1, b2);
      M3D V;
      V.col(0) = ray;
      V.col(1) = b1;
      V.col(2) = b2;

      M3D A_inv = M3D::Zero();
      A_inv(0, 0) = range_var_inv;
      A_inv(1, 1) = tangent_var_inv;
      A_inv(2, 2) = tangent_var_inv;

      return V * A_inv * V.transpose();
  }
} pointWithCov;

typedef struct Plane {
    Plane() : points_size(0) , radius(0)
    {
        center = Eigen::Vector3d::Zero();
        normal = Eigen::Vector3d::Zero();
        y_normal = Eigen::Vector3d::Zero();
        x_normal = Eigen::Vector3d::Zero();
        covariance = Eigen::Matrix3d::Zero();
        plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    }

    void reset_params()
    {
        plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
        covariance = Eigen::Matrix3d::Zero();
        center = Eigen::Vector3d::Zero();
        normal = Eigen::Vector3d::Zero();
        radius = 0;
    }
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d y_normal; // mid
  Eigen::Vector3d x_normal; // max
  Eigen::Matrix3d covariance;
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d = 0;
  int points_size = 0;

  int points_size_ncov = 0;
  bool is_ncov_init = false;
  vector<double> lambda_cov;

  bool is_plane = false;
  bool is_init = false;
  int id;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;
  bool update_enable = true;
} Plane;

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

M3D calcIncidentCov(const V3D & lidar2p, const V3D& normal)
{
    static double angle_rad = DEG2RAD(angle_cov);
    V3D ray = lidar2p.normalized();
    double cov_scale = calcIncidentCovScale(ray, lidar2p.norm(), normal); // ^2
    return cov_scale * ray * ray.transpose();
}

M3D calcIncidentCov(const pointWithCov & pv, const V3D& normal)
{
    static double angle_rad = DEG2RAD(angle_cov);
    const V3D & ray = pv.ray;
    double cov_scale = calcIncidentCovScale(ray, pv.p2lidar, normal); // ^2
    return cov_scale * ray * ray.transpose();
}

double calcRoughCovScale(const float& r)
{
//    if (r < 0.01)
//        return 0.0;
    double rou_cov = roughness_cov_scale * r * r;
    return min(rou_cov, roughness_cov_max);
}

M3D calcRoughCov(const float & roughness, const Eigen::Vector3d & p)
{
    V3D ray = p.normalized();
    double r = calcRoughCovScale(roughness);
    return r * ray * ray.transpose();
}

// PCA Eigen Solver
void PCAEigenSolver(const M3D & covariance, M3D & eigen_vectors, V3D & eigen_values)
{
    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    eigen_vectors.col(0) = evecs.real().col(evalsMin);
    eigen_vectors.col(1) = evecs.real().col(evalsMid);
    eigen_vectors.col(2) = evecs.real().col(evalsMax);
    eigen_values(0) = evalsReal[evalsMin];
    eigen_values(1) = evalsReal[evalsMid];
    eigen_values(2) = evalsReal[evalsMax];
}

void calcLambdaCov(const vector<pointWithCov> & points, const vector<M3D> & Jpi, vector<double> & lambda_cov)
{
    ROS_ASSERT(points.size() == Jpi.size());
    lambda_cov.resize(3, 0.0);
    for (int i = 0; i < points.size(); ++i) {
        lambda_cov[0] += Jpi[i].row(0) * points[i].cov * Jpi[i].row(0).transpose();
        lambda_cov[1] += Jpi[i].row(1) * points[i].cov * Jpi[i].row(1).transpose();
        lambda_cov[2] += Jpi[i].row(2) * points[i].cov * Jpi[i].row(2).transpose();
    }
}

void calcLambdaCovIncremental(const vector<pointWithCov > & points, const vector<M3D> & Jpi,
                              const vector<double> & lambda_cov_old, vector<double> & lambda_cov_incre)
{
//    printf("calcLambdaCovIncremental\n");
//    assert(points.size() == Jpi.size());
    lambda_cov_incre.resize(3);
    double n = (double)points.size();
    double scale = pow((n - 1.0), 2) / (n * n); //^2
    for (int i = 0; i < 3; ++i) {
        lambda_cov_incre[i] = lambda_cov_old[i] * scale +
                              Jpi.back().row(i) * points.back().cov * Jpi.back().row(i).transpose();
    }
}

void JacobianLambda(const vector<pointWithCov> & points, const M3D & eigen_vectors,
                    const V3D & center, vector<M3D> & Jpi)
{
    double n = (double)points.size();
    Jpi.resize(points.size());
    vector<M3D> uk_ukt(3);
    for (int i = 0; i < 3; ++i) {
        uk_ukt[i] = 2.0 / n * eigen_vectors.col(i) * eigen_vectors.col(i).transpose(); // 2 / n * v_k * v_k^t
    }
    double cov_lambda1 = 0, cov_lambda2 = 0, cov_lambda3 = 0;
    for (int i = 0; i < points.size(); ++i) {
        V3D center2point = points[i].point - center;
        Jpi[i].row(0) = center2point.transpose() * uk_ukt[0];
        Jpi[i].row(1) = center2point.transpose() * uk_ukt[1];
        Jpi[i].row(2) = center2point.transpose() * uk_ukt[2];
//
//        cov_lambda1 += Jpi[i].row(0) * M3D::Identity() * Jpi[i].row(0).transpose();
//        cov_lambda2 += Jpi[i].row(1) * M3D::Identity() * Jpi[i].row(1).transpose();
//        cov_lambda3 += Jpi[i].row(2) * M3D::Identity() * Jpi[i].row(2).transpose();
    }
}

double incrementalJacobianLambda(const vector<pointWithCov> & points, const M3D & eigen_vectors_old, const V3D & center_old,
                                 const M3D & eigen_vectors_new, const V3D & center_new, vector<M3D> & Jpi_incre)
{
    double n = (double)points.size();
    const V3D & xn = points.back().point;
    V3D xn_mn1 = xn - center_old;
//    vector<V3D> en(3), en_1(3); // eigen vectors of n, n-1
//    vector<double> cos_en(3), cos_en_1(3), cos_theta(3);
    vector<V3D> term_2(3);
//    printf("derivative increment:\n");
    for (int i = 0; i < 3; ++i) {
        const V3D & en = eigen_vectors_new.col(i); // n
        const V3D & en_1 = eigen_vectors_old.col(i); // n - 1

        double cos_en = abs(xn_mn1.dot(en)); // d * mx_mn1
        double cos_en_1 = abs(xn_mn1.dot(en_1));
        double cos_theta = abs(en.dot(en_1));
        term_2[i] =  (cos_en * en_1 + cos_en_1 * en_1) / (n * n * cos_theta);
//        printf("lambda increment term %d: %e %e %e\n", i + 1, term_2[i](0), term_2[i](1), term_2[i](2));
    }
//    Jpi_incre.resize(points.size());
//    double scale_1 = (n - 1) / n;
    // todo test * n-1
//    if (term_2[0].norm() * (n - 1.0) > lambda_cov_threshold)
    if (term_2[0].norm() > lambda_cov_threshold)
    {
        JacobianLambda(points, eigen_vectors_new, center_new, Jpi_incre);
//        for (int i = 0; i < points.size() - 1; ++i) { // i = 1, 2 ..., n-1
//            Jpi_incre[i].row(0) = scale_1 * Jpi_old[i].row(0) - term_2[0].transpose(); // d lambda1 d p
//            Jpi_incre[i].row(1) = scale_1 * Jpi_old[i].row(1) - term_2[1].transpose(); // d lambda2 d p
//            Jpi_incre[i].row(2) = scale_1 * Jpi_old[i].row(2) - term_2[2].transpose(); // d lambda3 d p
//        }
    }
    else {
        Jpi_incre.resize(1);
        // for i = n
        Jpi_incre.back().row(0) = term_2[0].transpose() * (n - 1); // d lambda1 d p
        Jpi_incre.back().row(1) = term_2[1].transpose() * (n - 1); // d lambda2 d p
        Jpi_incre.back().row(2) = term_2[2].transpose() * (n - 1); // d lambda3 d p
    }
    return term_2[0].norm();
}


void calcNormalCov(const vector<pointWithCov > & points, const M3D & eigen_vectors, const V3D & eigen_values,
                   const V3D& center, M6D & plane_cov)
{
    const V3D & evecMin = eigen_vectors.col(0);
    const V3D & evecMid = eigen_vectors.col(1);
    const V3D & evecMax = eigen_vectors.col(2);

    int points_size = points.size();
    plane_cov = M6D::Zero();
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / points_size, 0, 0, 0, 1.0 / points_size, 0, 0, 0,
            1.0 / points_size;
    for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F = M3D::Zero();
        V3D p_center = points[i].point - center;
        F.row(1)  = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(1))) *
                    (evecMid * evecMin.transpose() + evecMin * evecMid.transpose());
        F.row(2) = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(2))) *
                   (evecMax * evecMin.transpose() + evecMin * evecMax.transpose());

        J.block<3, 3>(0, 0) = eigen_vectors * F;
        J.block<3, 3>(3, 0) = J_Q;

        if (points.size() < normal_cov_incre_min)
            plane_cov += J * points[i].cov * J.transpose();
        else {
            const pointWithCov &pv = points[i];
            double cos_theta = evecMin.dot(pv.normal.cast<double>()); // [0, 1.0]
            double roughness = (1 - cos_theta * cos_theta) * roughness_cov_scale; // sin^2_theta * roughness
            M3D point_cov = pv.cov + roughness * M3D::Identity() +
                            calcIncidentCovScale(pv.ray, pv.p2lidar, evecMin) * pv.ray * pv.ray.transpose();
            plane_cov += J * point_cov * J.transpose();
        }
    }
}

double calcNormalCovIncremental(const vector<pointWithCov > & points, const M3D & eigen_vectors_old,
                                const V3D & eigen_values_old, const V3D& center_old, const M6D & nq_cov_old,
                                const M3D & eigen_vectors_new, const V3D & eigen_values_new, M6D & plane_cov)
{
    double n = (double)points.size();

    const V3D & Vk_min = eigen_vectors_new.col(0); // eigen vector of n points covirance
    const V3D & Vk_mid = eigen_vectors_new.col(1);
    const V3D & Vk_max = eigen_vectors_new.col(2);
    const V3D & Vk1_min = eigen_vectors_old.col(0); // eigen vector of n-1 points covirance
    const V3D & Vk1_mid = eigen_vectors_old.col(1);
    const V3D & Vk1_max = eigen_vectors_old.col(2);

    const V3D & xn = points.back().point;
    V3D xn_mn1 = xn - center_old;

    double cos_min = abs(xn_mn1.dot(Vk_min)); // vector mid * mx_mn1
    double cos_mid = abs(xn_mn1.dot(Vk_mid)); // vector mid * mx_mn1
    double cos_max = abs(xn_mn1.dot(Vk_max));
    double lambda_k_min_mid = eigen_values_new(0) - eigen_values_new(1); // n points
    double lambda_k_min_max = eigen_values_new(0) - eigen_values_new(2);
    double lambda_k1_min_mid = eigen_values_old(0) - eigen_values_old(1); // n-1 points
    double lambda_k1_min_max = eigen_values_old(0) - eigen_values_old(2);
    double scale_1 = (n - 1.0) / n * abs(Vk_min.dot(Vk1_min));
//    printf("scale_1 %f\n", scale_1);

//    // todo test * n-1
//    V3D term_2_mid = (cos_mid * Vk_min + cos_min * Vk_mid) / (n * n * lambda_k_min_mid) * (n - 1.0);
//    V3D term_2_max = (cos_max * Vk_min + cos_min * Vk_max) / (n * n * lambda_k_min_max) * (n - 1.0);
    V3D term_2_mid = (cos_mid * Vk_min + cos_min * Vk_mid) / (n * n * lambda_k_min_mid);
    V3D term_2_max = (cos_max * Vk_min + cos_min * Vk_max) / (n * n * lambda_k_min_max);
//    printf("term_2 magnitude mid %e max %e\n", term_2_mid.norm(), term_2_max.norm());
    double diff_magnitude = abs(term_2_mid.norm() + term_2_max.norm());
    if (diff_magnitude > normal_cov_threshold)
        return diff_magnitude;

    double scale_mid = lambda_k1_min_mid / lambda_k_min_mid * abs(Vk_mid.dot(Vk1_mid));
    double scale_max = lambda_k1_min_max / lambda_k_min_max * abs(Vk_max.dot(Vk1_max));

    int points_size = points.size();
    plane_cov = M6D::Zero();

    M3D scale_matrix = eigen_vectors_old.transpose();
    scale_matrix.row(1) *= scale_mid;
    scale_matrix.row(2) *= scale_max;
    scale_matrix = eigen_vectors_new * scale_matrix;
    plane_cov.block<3, 3>(0, 0) = pow(scale_1, 2) * scale_matrix * nq_cov_old.block<3, 3>(0, 0) * scale_matrix.transpose();

    M3D top_right = scale_1 * scale_matrix * nq_cov_old.block<3, 3>(0, 3) * (n - 1) / n;
    plane_cov.block<3, 3>(0, 3) = top_right;
    plane_cov.block<3, 3>(3, 0) = top_right.transpose();

    plane_cov.block<3, 3>(3, 3) = nq_cov_old.block<3, 3>(3, 3) * pow((n - 1.0), 2) / (n * n);

    /// for the new point
    double scale_n = (n - 1.0) / (n * n);
    V3D term_n_mid = scale_n / lambda_k_min_mid * (cos_mid * Vk_min + cos_min * Vk_mid);
    V3D term_n_max = scale_n / lambda_k_min_max * (cos_max * Vk_min + cos_min * Vk_max);
    M3D J_n = M3D::Zero();
    J_n.row(1) = term_n_mid.transpose();
    J_n.row(2) = term_n_max.transpose();
    J_n = eigen_vectors_new * J_n;

    M3D point_cov = points.back().cov;
    if (points.size() >= normal_cov_incre_min) {
        const pointWithCov &pv = points.back();
        double cos_theta = abs(Vk_min.dot(pv.normal.cast<double>())); // [0, 1.0]
//        double sin_theta2 = 1 - cos_theta * cos_theta;
        double roughness = pow((1 - cos_theta), 2) * roughness_cov_scale;
        point_cov = pv.cov + roughness * M3D::Identity() +
                        calcIncidentCovScale(pv.ray, pv.p2lidar, Vk_min) * pv.ray * pv.ray.transpose();
    }


    plane_cov.block<3, 3>(0, 0) += J_n * point_cov * J_n.transpose();

    M3D top_right_n = J_n * point_cov / n;
    plane_cov.block<3, 3>(0, 3) += top_right_n;
    plane_cov.block<3, 3>(3, 0) += top_right_n.transpose();

    plane_cov.block<3, 3>(3, 3) += point_cov / (n * n);

    return diff_magnitude;
}

class OctoTree {
public:
  std::vector<pointWithCov> temp_points_; // all points in an octo tree
  std::vector<pointWithCov> new_points_;  // new points in an octo tree
  Plane *plane_ptr_;
  int max_layer_;
  bool indoor_mode_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  int max_cov_points_size_;
  bool init_octo_;
  bool update_cov_enable_;
  bool update_enable_;
  V3F ringfals_normal;
  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size, int max_cov_points_size, float planer_threshold)
      : max_layer_(max_layer), layer_(layer),
        layer_point_size_(layer_point_size), max_points_size_(max_point_size),
        max_cov_points_size_(max_cov_points_size),
        planer_threshold_(planer_threshold) {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_num_ = 0;
    all_points_num_ = 0;
    // when new points num > 5, do a update
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    update_cov_enable_ = true;
    max_plane_update_threshold_ = layer_point_size_[layer_];
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }

  void downsampleNormal(const std::vector<pointWithCov> &points) {
      if (points.empty())
          return;
      ringfals_normal = V3F::Zero();
      for (auto pv : points)
          ringfals_normal += pv.normal;
      ringfals_normal.normalize();
  }
    void downsampleNormal() {
        if (temp_points_.empty())
            return;
        ringfals_normal = V3F::Zero();
        for (auto pv : temp_points_)
            ringfals_normal += pv.normal;
        ringfals_normal.normalize();
    }
    void downsampleNormalIncre() {
        if (new_points_.empty())
            return;
        Eigen::Vector3f new_ringfals = Eigen::Vector3f::Zero();
        for (auto pv : new_points_)
            new_ringfals += pv.normal;
        ringfals_normal = new_ringfals + ringfals_normal * temp_points_.size();
        ringfals_normal.normalize();
    }


    double point2planeResidual(const vector<pointWithCov> & points, const V3D & centroid, const V3D & normal)
    {
        double sum_dist2 = 0;
        for (int i = 0; i < points.size(); ++i) {
            double dist = normal.dot(points[i].point - centroid);
            sum_dist2 += dist * dist;
        }
        return sum_dist2;
    }

    double diff_normal(const V3D & n1, const V3D & n2)
    {
        V3D x_cross_y = n1.cross(n2);
        double x_dot_y = n1.dot(n2);
        double theta = atan2(x_cross_y.norm(), abs(x_dot_y));
        return theta;
    }

  void init_plane(const std::vector<pointWithCov> &points, Plane *plane) {
        double t_cov;
      int num_points_old = plane->points_size;
      V3D center_old = plane->center;

        plane->points_size = points.size();
        double t1 = omp_get_wtime();
      if (num_points_old > 10)
      {
          int m = points.size();
          int iter_points = m - num_points_old;
          for (int i = m - iter_points; i < m; ++i) {
              double n = i + 1;
              const V3D & xn = points[i].point;
              V3D xn_mn_1 = xn - plane->center;
              plane->covariance = (n - 1) / n * (plane->covariance + (xn_mn_1 * xn_mn_1.transpose()) / n);
              plane->center = plane->center / n * (n - 1) + xn / n;
          }
      }
      else
      {
        plane->reset_params();
        for (int i = 0; i < plane->points_size; ++i) {
            const pointWithCov &pv = points[i];
            plane->covariance += pv.point * pv.point.transpose();
            plane->center += pv.point;
        }
        plane->center = plane->center / plane->points_size;
        plane->covariance = plane->covariance / plane->points_size - plane->center * plane->center.transpose();
      }
        t_cov = omp_get_wtime() - t1;

      // record old parameters
      M3D eigen_vectors_old;
      eigen_vectors_old.col(0) = plane->normal;
      eigen_vectors_old.col(1) = plane->y_normal;
      eigen_vectors_old.col(2) = plane->x_normal;
      V3D eigen_values_old(plane->min_eigen_value, plane->mid_eigen_value, plane->max_eigen_value);
      M6D n_q_cov_old = plane->plane_cov;
      vector<double> lambda_cov_old = plane->lambda_cov;

      // PCA
      M3D eigen_vectors_new;
      V3D eigen_values_new;
      PCAEigenSolver(plane->covariance, eigen_vectors_new, eigen_values_new);
//      V3D normal = eigen_vectors_new.col(0);
//      if (normal.cast<float>().dot(ringfals_normal) < 0)
//          eigen_vectors_new.col(0) = -normal;

      // plane covariance calculation
    if (eigen_values_new(0) < planer_threshold_) {
        // cov should be init first, plane->is_plane = true;
        bool update_cov_std = !plane->is_ncov_init || plane->points_size < normal_cov_incre_min ||
                plane->points_size == max_cov_points_size_ || plane->points_size == max_points_size_ ||
                plane->points_size - plane->points_size_ncov >= normal_cov_update_interval;

        if (update_cov_std) {
            // lambda
            vector<M3D> J_lambda_incre;
            JacobianLambda(points, eigen_vectors_new, plane->center, J_lambda_incre);
            calcLambdaCov(points, J_lambda_incre, plane->lambda_cov);

            // normal center
            calcNormalCov(points, eigen_vectors_new, eigen_values_new, plane->center, plane->plane_cov);
            plane->points_size_ncov = plane->points_size;
            plane->is_ncov_init = true;
        }
        else {
            // check lambda increment first
            vector<M3D> J_lambda_incre;
//        vector<double> lambda_cov_incre;
            double d_lambda0 = incrementalJacobianLambda(points, eigen_vectors_old, center_old,
                                                         eigen_vectors_new, plane->center, J_lambda_incre);
            double normal_cov_diff;
            normal_cov_diff =
                    calcNormalCovIncremental(points, eigen_vectors_old, eigen_values_old, center_old,
                                             n_q_cov_old, eigen_vectors_new, eigen_values_new, plane->plane_cov);

            if (d_lambda0 <= lambda_cov_threshold && normal_cov_diff <= normal_cov_threshold) {
                calcLambdaCovIncremental(points, J_lambda_incre, lambda_cov_old, plane->lambda_cov);
            } else { //update lambda, normal, center covariance in std form
                JacobianLambda(points, eigen_vectors_new, plane->center, J_lambda_incre);
                calcLambdaCov(points, J_lambda_incre, plane->lambda_cov);
                calcNormalCov(points, eigen_vectors_new, eigen_values_new, plane->center, plane->plane_cov);
                plane->points_size_ncov = plane->points_size;
            }
        }

        plane->normal = eigen_vectors_new.col(0);
        plane->y_normal = eigen_vectors_new.col(1);
        plane->x_normal = eigen_vectors_new.col(2);
        plane->min_eigen_value = eigen_values_new(0);
        plane->mid_eigen_value = eigen_values_new(1);
        plane->max_eigen_value = eigen_values_new(2);
        plane->radius = sqrt(eigen_values_new(2));
        plane->d = -(plane->normal.dot(plane->center));
        plane->is_plane = true;

      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }

      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }

    } else {
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      plane->is_plane = false;
        plane->normal = eigen_vectors_new.col(0);
        plane->y_normal = eigen_vectors_new.col(1);
        plane->x_normal = eigen_vectors_new.col(2);
        plane->min_eigen_value = eigen_values_new(0);
        plane->mid_eigen_value = eigen_values_new(1);
        plane->max_eigen_value = eigen_values_new(2);
        plane->radius = sqrt(eigen_values_new(2));
        plane->d = -(plane->normal.dot(plane->center));
    }
  }

  // only updaye plane normal, center and radius with new points
  void update_plane(const std::vector<pointWithCov> &points, Plane *plane) {
    Eigen::Matrix3d old_covariance = plane->covariance;
    Eigen::Vector3d old_center = plane->center;
    Eigen::Matrix3d sum_ppt =
        (plane->covariance + plane->center * plane->center.transpose()) *
        plane->points_size;
    Eigen::Vector3d sum_p = plane->center * plane->points_size;
    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Vector3d pv = points[i].point;
      sum_ppt += pv * pv.transpose();
      sum_p += pv;
    }
    plane->points_size = plane->points_size + points.size();
    plane->center = sum_p / plane->points_size;
    plane->covariance = sum_ppt / plane->points_size -
                        plane->center * plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    if (evalsReal(evalsMin) < planer_threshold_) {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      if (ringfals_normal.dot(plane->normal.cast<float>()) < 0)
          plane->normal *= -1;
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));

      plane->is_plane = true;
      plane->is_update = true;
    } else {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      if (ringfals_normal.dot(plane->normal.cast<float>()) < 0)
          plane->normal *= -1;
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = false;
      plane->is_update = true;
    }
  }

  void init_octo_tree() {
      downsampleNormal(temp_points_);
      if (temp_points_.size() > max_plane_update_threshold_) {
        init_plane(temp_points_, plane_ptr_);
      if (plane_ptr_->is_plane == true) {
        octo_state_ = 0;
        if (temp_points_.size() > max_cov_points_size_) {
          update_cov_enable_ = false;
        }
        if (temp_points_.size() > max_points_size_) {
          update_enable_ = false;
        }
      } else {
        octo_state_ = 1;
        cut_octo_tree();
      }
      init_octo_ = true;
      new_points_num_ = 0;
      //      temp_points_.clear();
    }
  }

  void cut_octo_tree() {
    if (layer_ >= max_layer_) {
      octo_state_ = 0;
      return;
    }
    for (size_t i = 0; i < temp_points_.size(); i++) {
      int xyz[3] = {0, 0, 0};
      if (temp_points_[i].point[0] > voxel_center_[0]) {
        xyz[0] = 1;
      }
      if (temp_points_[i].point[1] > voxel_center_[1]) {
        xyz[1] = 1;
      }
      if (temp_points_[i].point[2] > voxel_center_[2]) {
        xyz[2] = 1;
      }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(
            max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
            max_cov_points_size_, planer_threshold_);
        leaves_[leafnum]->voxel_center_[0] =
            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      }
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_num_++;
    }
    for (uint i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        if (leaves_[i]->temp_points_.size() >
            leaves_[i]->max_plane_update_threshold_) {
//            downsampleNormal(temp_points_);
            init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
          if (leaves_[i]->plane_ptr_->is_plane) {
            leaves_[i]->octo_state_ = 0;
          } else {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_num_ = 0;
        }
          leaves_[i]->downsampleNormal();
      }
    }
  }

  void UpdateOctoTree(const pointWithCov &pv) {
    if (!init_octo_) {
      new_points_num_++;
      all_points_num_++;
      temp_points_.push_back(pv);
      downsampleNormal(temp_points_);
      if (temp_points_.size() > max_plane_update_threshold_) {
        init_octo_tree();
      }
    } else {
        if (plane_ptr_->is_plane) {
        if (update_enable_) {
          new_points_num_++;
          all_points_num_++;
          if (update_cov_enable_) {
            temp_points_.push_back(pv);
            downsampleNormal(temp_points_);
          } else {
            new_points_.push_back(pv);
            downsampleNormalIncre();
          }
          if (new_points_num_ > update_size_threshold_) {
            if (update_cov_enable_) {
                init_plane(temp_points_, plane_ptr_);
            }
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_cov_points_size_) {
            update_cov_enable_ = false;
            std::vector<pointWithCov>().swap(temp_points_);
          }
          if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
            plane_ptr_->update_enable = false;
            std::vector<pointWithCov>().swap(new_points_);
          }
        } else {
//            downsampleNormal(temp_points_); // todo comment this line
            return;
        }
      } else {
        if (layer_ < max_layer_) {
          if (temp_points_.size() != 0) {
            std::vector<pointWithCov>().swap(temp_points_);
          }
          if (new_points_.size() != 0) {
            std::vector<pointWithCov>().swap(new_points_);
          }
          int xyz[3] = {0, 0, 0};
          if (pv.point[0] > voxel_center_[0]) {
            xyz[0] = 1;
          }
          if (pv.point[1] > voxel_center_[1]) {
            xyz[1] = 1;
          }
          if (pv.point[2] > voxel_center_[2]) {
            xyz[2] = 1;
          }
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          if (leaves_[leafnum] != nullptr) {
            leaves_[leafnum]->UpdateOctoTree(pv);
          } else {
            leaves_[leafnum] = new OctoTree(
                max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                max_cov_points_size_, planer_threshold_);
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] =
                voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] =
                voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] =
                voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            leaves_[leafnum]->UpdateOctoTree(pv);
          }
        } else {
          if (update_enable_) {
            new_points_num_++;
            all_points_num_++;
            if (update_cov_enable_) {
              temp_points_.push_back(pv);
                downsampleNormal(temp_points_);
            } else {
              new_points_.push_back(pv);
                downsampleNormalIncre();
            }
            if (new_points_num_ > update_size_threshold_) {
              if (update_cov_enable_) {
                  init_plane(temp_points_, plane_ptr_);
              } else {
                update_plane(new_points_, plane_ptr_);
                new_points_.clear();
              }
              new_points_num_ = 0;
            }
            if (all_points_num_ >= max_cov_points_size_) {
              update_cov_enable_ = false;
              std::vector<pointWithCov>().swap(temp_points_);
            }
            if (all_points_num_ >= max_points_size_) {
              update_enable_ = false;
              plane_ptr_->update_enable = false;
              std::vector<pointWithCov>().swap(new_points_);
            }
          }
        }
      }
    }
  }
};

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

void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
        //already have the voexl
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
      // init voxel and divide into sub voxel
    iter->second->init_octo_tree(); //OctoTree *
  }
}

void updateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                           (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            feat_map[position]->UpdateOctoTree(p_v);
        } else {
            OctoTree *octo_tree =
                    new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                 max_cov_points_size, planer_threshold);
            feat_map[position] = octo_tree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
            feat_map[position]->UpdateOctoTree(p_v);
        }
    }
}

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {

  std::unordered_map<VOXEL_LOC, vector<pointWithCov>> position_index_map;
  int insert_count = 0, update_count = 0;
  uint plsize = input_points.size();


  double t_update_start = omp_get_wtime();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      update_count++;
      position_index_map[position].push_back(p_v);
    } else {
      insert_count++;
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
  double t_update_end = omp_get_wtime();
  std::printf("Insert & store time:  %.4fs\n", t_update_end - t_update_start);
    t_update_start = omp_get_wtime();

#ifdef MP_EN
    omp_set_num_threads(num_update_thread);
#pragma omp parallel for default(none) shared(position_index_map, feat_map)
#endif
    for (size_t b = 0; b < position_index_map.bucket_count(); b++) {
        for (auto bi = position_index_map.begin(b); bi != position_index_map.end(b); bi++) {
            VOXEL_LOC position = bi->first;
            for (const pointWithCov &p_v:bi->second) {
                feat_map[position]->UpdateOctoTree(p_v);
            }
        }
    }
    t_update_end = omp_get_wtime();
    std::printf("Update:  %.4fs\n", t_update_end - t_update_start);

  std::printf("Insert: %d  Update: %d \n", insert_count, update_count);
}


void build_single_residual(const pointWithCov &pv, const OctoTree *current_octo,
                           const int current_layer, const int max_layer,
                           const double sigma_num, const V3F & body2point, bool &is_sucess,
                           double &prob, ptpl &single_ptpl) {
  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_world;
  if (current_octo->plane_ptr_->is_plane) {
      //check normal consistensy
      if (check_normal && body2point.dot(current_octo->plane_ptr_->normal.cast<float>()) > 0)
//      if (check_normal && body2point.dot(current_octo->ringfals_normal) > 0)
          return;

    Plane &plane = *current_octo->plane_ptr_;
    float dis_to_plane =
        fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
             plane.normal(2) * p_w(2) + plane.d);
    float dis_to_center =
        (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
        (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
        (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));

    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.point_world = pv.point_world;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
          single_ptpl.cov_lidar = pv.cov_lidar;
          single_ptpl.point_normal = pv.normal;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {

          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, max_layer,
                                sigma_num, body2point, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}

void GetUpdateNormal(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<pair<V3D, V3F>> &center_normal_list) {
    if (current_octo->layer_ > pub_max_voxel_layer) {
        return;
    }
//    if (!current_octo->init_octo_ || current_octo->temp_points_.size() == 0)
//        return;
//    if (current_octo->plane_ptr_->is_update && current_octo->ringfals_normal.norm() == 1) {
    if (abs(current_octo->ringfals_normal.norm() - 1) > 0.001) {
//        center_normal_list.push_back({current_octo->plane_ptr_->center, current_octo->ringfals_normal});
        ROS_ERROR("GetUpdateNormal norm: %f, points %d init %d", current_octo->ringfals_normal.norm(),
                  (int) current_octo->temp_points_.size(), current_octo->init_octo_);
    }
    if (current_octo->plane_ptr_->is_update && abs(current_octo->ringfals_normal.norm() - 1) < 0.001) {
        center_normal_list.push_back({current_octo->plane_ptr_->center, current_octo->ringfals_normal});
    }
    if (current_octo->layer_ < current_octo->max_layer_) {
        if (!current_octo->plane_ptr_->is_plane) {
            for (size_t i = 0; i < 8; i++) {
                if (current_octo->leaves_[i] != nullptr) {
                    GetUpdateNormal(current_octo->leaves_[i], pub_max_voxel_layer,
                                    center_normal_list);
                }
            }
        }
    }
    return;
}

// void BuildResidualListTBB(const unordered_map<VOXEL_LOC, OctoTree *>
// &voxel_map,
//                           const double voxel_size, const double sigma_num,
//                           const int max_layer,
//                           const std::vector<pointWithCov> &pv_list,
//                           std::vector<ptpl> &ptpl_list,
//                           std::vector<Eigen::Vector3d> &non_match) {
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<ptpl> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
//   std::for_each(
//       std::execution::par_unseq, index.begin(), index.end(),
//       [&](const size_t &i) {
//         pointWithCov pv = pv_list[i];
//         float loc_xyz[3];
//         for (int j = 0; j < 3; j++) {
//           loc_xyz[j] = pv.point_world[j] / voxel_size;
//           if (loc_xyz[j] < 0) {
//             loc_xyz[j] -= 1.0;
//           }
//         }
//         VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                            (int64_t)loc_xyz[2]);
//         auto iter = voxel_map.find(position);
//         if (iter != voxel_map.end()) {
//           OctoTree *current_octo = iter->second;
//           ptpl single_ptpl;
//           bool is_sucess = false;
//           double prob = 0;
//           build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                                 is_sucess, prob, single_ptpl);
//           if (!is_sucess) {
//             VOXEL_LOC near_position = position;
//             if (loc_xyz[0] > (current_octo->voxel_center_[0] +
//                               current_octo->quater_length_)) {
//               near_position.x = near_position.x + 1;
//             } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                      current_octo->quater_length_)) {
//               near_position.x = near_position.x - 1;
//             }
//             if (loc_xyz[1] > (current_octo->voxel_center_[1] +
//                               current_octo->quater_length_)) {
//               near_position.y = near_position.y + 1;
//             } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                      current_octo->quater_length_)) {
//               near_position.y = near_position.y - 1;
//             }
//             if (loc_xyz[2] > (current_octo->voxel_center_[2] +
//                               current_octo->quater_length_)) {
//               near_position.z = near_position.z + 1;
//             } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                      current_octo->quater_length_)) {
//               near_position.z = near_position.z - 1;
//             }
//             auto iter_near = voxel_map.find(near_position);
//             if (iter_near != voxel_map.end()) {
//               build_single_residual(pv, iter_near->second, 0, max_layer,
//                                     sigma_num, is_sucess, prob, single_ptpl);
//             }
//           }
//           if (is_sucess) {

//             mylock.lock();
//             useful_ptpl[i] = true;
//             all_ptpl_list[i] = single_ptpl;
//             mylock.unlock();
//           } else {
//             mylock.lock();
//             useful_ptpl[i] = false;
//             mylock.unlock();
//           }
//         }
//       });
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.push_back(all_ptpl_list[i]);
//     }
//   }
// }

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          const V3F & body_pos,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); i++) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);

    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      V3F body2point = pv.point_world.cast<float>() - body_pos;
      body2point.normalize();
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, max_layer, sigma_num, body2point,
                            is_sucess, prob, single_ptpl);
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num, body2point,
                                is_sucess, prob, single_ptpl);
        }
      }

      if (is_sucess) {

        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

//void BuildResidualListNormal(
//    const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
//    const double voxel_size, const double sigma_num, const int max_layer,
//    const std::vector<pointWithCov> &pv_list, std::vector<ptpl> &ptpl_list,
//    std::vector<Eigen::Vector3d> &non_match) {
//  ptpl_list.clear();
//  std::vector<size_t> index(pv_list.size());
//  for (size_t i = 0; i < pv_list.size(); ++i) {
//    pointWithCov pv = pv_list[i];
//    float loc_xyz[3];
//    for (int j = 0; j < 3; j++) {
//      loc_xyz[j] = pv.point_world[j] / voxel_size;
//      if (loc_xyz[j] < 0) {
//        loc_xyz[j] -= 1.0;
//      }
//    }
//    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                       (int64_t)loc_xyz[2]);
//    auto iter = voxel_map.find(position);
//    if (iter != voxel_map.end()) {
//      OctoTree *current_octo = iter->second;
//      ptpl single_ptpl;
//      bool is_sucess = false;
//      double prob = 0;
//      build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                            is_sucess, prob, single_ptpl);
//
//      if (!is_sucess) {
//        VOXEL_LOC near_position = position;
//        if (loc_xyz[0] >
//            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
//          near_position.x = near_position.x + 1;
//        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                 current_octo->quater_length_)) {
//          near_position.x = near_position.x - 1;
//        }
//        if (loc_xyz[1] >
//            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
//          near_position.y = near_position.y + 1;
//        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                 current_octo->quater_length_)) {
//          near_position.y = near_position.y - 1;
//        }
//        if (loc_xyz[2] >
//            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
//          near_position.z = near_position.z + 1;
//        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                 current_octo->quater_length_)) {
//          near_position.z = near_position.z - 1;
//        }
//        auto iter_near = voxel_map.find(near_position);
//        if (iter_near != voxel_map.end()) {
//          build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num,
//                                is_sucess, prob, single_ptpl);
//        }
//      }
//      if (is_sucess) {
//        ptpl_list.push_back(single_ptpl);
//      } else {
//        non_match.push_back(pv.point_world);
//      }
//    }
//  }
//}

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

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string plane_ns, const Plane &single_plane,
                    const float alpha, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center[0];
  plane.pose.position.y = single_plane.center[1];
  plane.pose.position.z = single_plane.center[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal, single_plane.y_normal,
                  single_plane.normal, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void pubNoPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                   const ros::Publisher &plane_map_pub) {
  int id = 0;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane) {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          OctoTree *temp_octo_tree = iter->second->leaves_[i];
          if (!temp_octo_tree->plane_ptr_->is_plane) {
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (!temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                  Eigen::Vector3d plane_rgb(1, 1, 1);
                  pubSinglePlane(voxel_plane, "no_plane",
                                 *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                 use_alpha, plane_rgb);
                }
              }
            }
          }
        }
      }
    }
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubVoxelMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<Plane> pub_plane_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                 const ros::Publisher &plane_map_pub) {
  OctoTree *current_octo = nullptr;

  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 1.0;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);

  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_update) {
      Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

      V3D plane_cov =
          iter->second->plane_ptr_->plane_cov.block<3, 3>(0, 0).diagonal();
      double trace = plane_cov.sum();
      if (trace >= max_trace) {
        trace = max_trace;
      }
      trace = trace * (1.0 / max_trace);
      trace = pow(trace, pow_num);
      uint8_t r, g, b;
      mapJet(trace, 0, 1, r, g, b);
      Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
      // Eigen::Vector3d plane_rgb(1, 0, 0);
      float alpha = 0.0;
      if (iter->second->plane_ptr_->is_plane) {
        alpha = use_alpha;
      } else {
        // std::cout << "delete plane" << std::endl;
      }
      // if (iter->second->update_enable_) {
      //   plane_rgb << 1, 0, 0;
      // } else {
      //   plane_rgb << 0, 0, 1;
      // }
      pubSinglePlane(voxel_plane, "plane", *(iter->second->plane_ptr_), alpha,
                     plane_rgb);

      iter->second->plane_ptr_->is_update = false;
    } else {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          if (iter->second->leaves_[i]->plane_ptr_->is_update) {
            Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

            V3D plane_cov = iter->second->leaves_[i]
                                ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                                .diagonal();
            double trace = plane_cov.sum();
            if (trace >= max_trace) {
              trace = max_trace;
            }
            trace = trace * (1.0 / max_trace);
            // trace = (max_trace - trace) / max_trace;
            trace = pow(trace, pow_num);
            uint8_t r, g, b;
            mapJet(trace, 0, 1, r, g, b);
            Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
            plane_rgb << 0, 1, 0;
            // fabs(iter->second->leaves_[i]->plane_ptr_->normal[0]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[1]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[2]);
            float alpha = 0.0;
            if (iter->second->leaves_[i]->plane_ptr_->is_plane) {
              alpha = use_alpha;
            } else {
              // std::cout << "delete plane" << std::endl;
            }
            pubSinglePlane(voxel_plane, "plane",
                           *(iter->second->leaves_[i]->plane_ptr_), alpha,
                           plane_rgb);
            // loop.sleep();
            iter->second->leaves_[i]->plane_ptr_->is_update = false;
            // loop.sleep();
          } else {
            OctoTree *temp_octo_tree = iter->second->leaves_[i];
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update) {
                  if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                    // std::cout << "subsubplane" << std::endl;
                    Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
                    V3D plane_cov =
                        temp_octo_tree->leaves_[j]
                            ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                            .diagonal();
                    double trace = plane_cov.sum();
                    if (trace >= max_trace) {
                      trace = max_trace;
                    }
                    trace = trace * (1.0 / max_trace);
                    // trace = (max_trace - trace) / max_trace;
                    trace = pow(trace, pow_num);
                    uint8_t r, g, b;
                    mapJet(trace, 0, 1, r, g, b);
                    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
                    plane_rgb << 0, 0, 1;
                    float alpha = 0.0;
                    if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                      alpha = use_alpha;
                    }

                    pubSinglePlane(voxel_plane, "plane",
                                   *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                   alpha, plane_rgb);
                    // loop.sleep();
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  plane_map_pub.publish(voxel_plane);
  // plane_map_pub.publish(voxel_norm);
  loop.sleep();
  // cout << "[Map Info] Plane counts:" << plane_count
  //      << " Sub Plane counts:" << sub_plane_count
  //      << " Sub Sub Plane counts:" << sub_sub_plane_count << endl;
  // cout << "[Map Info] Update plane counts:" << update_count
  //      << "total size: " << feat_map.size() << endl;
}

M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc)
{
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  float tangent_var = pow(DEG2RAD(degree_inc), 2) * range * range; // d^2 * sigma^2

    Eigen::Vector3d direction(pb);
    direction.normalize();
    M3D rrt = direction * direction.transpose(); // ray * ray^t
  return range_var * rrt + tangent_var * (Eigen::Matrix3d::Identity() - rrt);

  Eigen::Matrix2d direction_var;
    // (angle_cov^2, 0,
    //  0, angle_cov^2)
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);

  Eigen::Matrix3d direction_hat; // w^
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  //direction dot base_vector1 = 0
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2)); //(1, 1, -(x+y)/z), not unique
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N; //N = [base_vector1, base_vector2]
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N; // (d * w^ * N )in the paper
  //cov = w * var_d * w^T + A * var_w * A^T
  return direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
};

double len = 1.0;
void pubMapRingFalsNormal(const std::unordered_map<VOXEL_LOC, OctoTree *> & voxel_map,
                          const int pub_max_voxel_layer, const ros::Time time,
                          const ros::Publisher & pub)
{
    double max_trace = 0.25;
    double pow_num = 0.2;
    ros::Rate loop(500);
    float use_alpha = 0.8;
    visualization_msgs::Marker normals;
    normals.header.stamp = time;
    normals.header.frame_id = "camera_init";
    normals.ns = "RingFals normal";
    normals.type = visualization_msgs::Marker::LINE_LIST;
    normals.action = visualization_msgs::Marker::ADD;
    normals.lifetime = ros::Duration();
    normals.pose.orientation.w = 1.0;
    normals.id = 0;
    normals.scale.x = 0.03;
    normals.scale.y = 0.03;
    normals.scale.z = 0.03;
    normals.color.g = 1.0;
    normals.color.a = 1.0;

    std::vector<pair<V3D, V3F>> center_normal_list;
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        GetUpdateNormal(iter->second, pub_max_voxel_layer, center_normal_list);
    }
    for (size_t i = 0; i < center_normal_list.size(); i++) {
        geometry_msgs::Point p;
        p.x = center_normal_list[i].first.x(); // center
        p.y = center_normal_list[i].first.y();
        p.z = center_normal_list[i].first.z();
        normals.points.push_back(p);

        p.x += center_normal_list[i].second.x() * len; // normal direction
        p.y += center_normal_list[i].second.y() * len;
        p.z += center_normal_list[i].second.z() * len;
        normals.points.push_back(p);
    }
    pub.publish(normals);
}

void pubMapVoxelNormal(const std::unordered_map<VOXEL_LOC, OctoTree *> & voxel_map,
                          const int pub_max_voxel_layer, const ros::Time time,
                          const ros::Publisher & pub)
{
    double max_trace = 0.25;
    double pow_num = 0.2;
    ros::Rate loop(500);
    float use_alpha = 0.8;
    visualization_msgs::Marker normals;
    normals.header.stamp = time;
    normals.header.frame_id = "camera_init";
    normals.ns = "voxelMap normal";
    normals.type = visualization_msgs::Marker::LINE_LIST;
    normals.action = visualization_msgs::Marker::ADD;
    normals.lifetime = ros::Duration();
    normals.pose.orientation.w = 1.0;
    normals.id = 0;
    normals.scale.x = 0.03;
    normals.scale.y = 0.03;
    normals.scale.z = 0.03;
    normals.color.b = 1.0;
    normals.color.g = 0.3;
    normals.color.a = 1.0;

    std::vector<Plane> pub_plane_list;
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
    }

    for (size_t i = 0; i < pub_plane_list.size(); i++) {
        geometry_msgs::Point p;
        p.x = pub_plane_list[i].center.x(); // center
        p.y = pub_plane_list[i].center.y();
        p.z = pub_plane_list[i].center.z();
        normals.points.push_back(p);

        p.x += pub_plane_list[i].normal.x() * len; // normal direction
        p.y += pub_plane_list[i].normal.y() * len;
        p.z += pub_plane_list[i].normal.z() * len;
        normals.points.push_back(p);
    }
    pub.publish(normals);
}

void pubScanWithCov(const ros::Publisher& publisher, const vector<pointWithCov> & cloud,
                    const string & frame_id= "lidar", const bool incident_and_Rohgh = false) {
    visualization_msgs::MarkerArray cloud_sphere;
    cloud_sphere.markers.reserve(100000);

    int point_size = cloud.size();

    for (int i = 0; i < point_size; ++i) {
        const pointWithCov & pv = cloud[i];

        visualization_msgs::Marker point_cov;
        point_cov.header.frame_id = frame_id;
        point_cov.ns = "cloud_cov";
        if (incident_and_Rohgh)
            point_cov.ns = "cloud_cov_ir";
        point_cov.id = i;
        point_cov.type = visualization_msgs::Marker::SPHERE;
        point_cov.action = visualization_msgs::Marker::ADD;
        point_cov.pose.position.x = pv.point(0);
        point_cov.pose.position.y = pv.point(1);
        point_cov.pose.position.z = pv.point(2);

        M3D rot;
        // find base vector in the local tangent space
        V3D bn1, bn2;
        findLocalTangentBases(pv.ray.cast<double>(), bn1, bn2);
        double range_cov, tan_cov;
        range_cov = ranging_cov * ranging_cov;
        tan_cov = pv.tangent_cov;
        if (incident_and_Rohgh) {
            double incident_scale = calcIncidentCovScale(pv.ray, pv.p2lidar, pv.normal.cast<double>());
//            range_cov = ranging_cov * ranging_cov + pv.roughness_cov + incident_scale;
            range_cov = ranging_cov * ranging_cov + incident_scale;
            tan_cov = pv.tangent_cov;
        }
        rot.block<3, 1>(0, 0) = pv.ray;
        rot.block<3, 1>(0, 1) = bn1;
        rot.block<3, 1>(0, 2) = bn2;
        Eigen::Quaterniond quat(rot);

        point_cov.pose.orientation.x = quat.x();
        point_cov.pose.orientation.y = quat.y();
        point_cov.pose.orientation.z = quat.z();
        point_cov.pose.orientation.w = quat.w();
        point_cov.scale.x = range_cov * visual_ray_scale;
        point_cov.scale.y = tan_cov * visual_tan_scale;
        point_cov.scale.z = tan_cov * visual_tan_scale;
        point_cov.color.a = visual_a_scale;
        point_cov.color.r = 1.0;
        point_cov.color.g = 0.0;
        point_cov.color.b = 0.0;
        if (incident_and_Rohgh)
        {
            point_cov.color.r = 0.0;
            point_cov.color.g = 1.0;
            point_cov.color.b = 0.0;
        }
        point_cov.lifetime = ros::Duration();
        cloud_sphere.markers.push_back(point_cov);
    }
    publisher.publish(cloud_sphere);
}


void pubScanRoughness(const ros::Publisher& publisher, const PointCloudXYZI::Ptr & cloud,
                    const string & frame_id= "lidar") {
    visualization_msgs::MarkerArray cloud_sphere;
    cloud_sphere.markers.reserve(100000);

    int point_size = cloud->size();

    for (int i = 0; i < point_size; ++i) {
        const PointType & p = cloud->points[i];

        visualization_msgs::Marker point_cov;
        point_cov.header.frame_id = frame_id;
        point_cov.ns = "roughness_cov";
//        if (incident_and_Rohgh)
//            point_cov.ns = "cloud_cov_ir";
        point_cov.id = i;
        point_cov.type = visualization_msgs::Marker::SPHERE;
        point_cov.action = visualization_msgs::Marker::ADD;
        point_cov.pose.position.x = p.x;
        point_cov.pose.position.y = p.y;
        point_cov.pose.position.z = p.z;

//        M3D rot;
//        // find base vector in the local tangent space
//        V3D bn1, bn2;
//        findLocalTangentBases(pv.ray.cast<double>(), bn1, bn2);
//        double range_cov, tan_cov;
//        range_cov = ranging_cov * ranging_cov;
//        tan_cov = pv.tangent_cov;
//        if (incident_and_Rohgh) {
//            double incident_scale = calcIncidentCovScale(pv.ray, pv.p2lidar, pv.normal.cast<double>());
//            range_cov = ranging_cov * ranging_cov + pv.roughness_cov + incident_scale;
//            tan_cov = pv.tangent_cov;
//        }
//        rot.block<3, 1>(0, 0) = pv.ray;
//        rot.block<3, 1>(0, 1) = bn1;
//        rot.block<3, 1>(0, 2) = bn2;
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

        point_cov.pose.orientation.x = quat.x();
        point_cov.pose.orientation.y = quat.y();
        point_cov.pose.orientation.z = quat.z();
        point_cov.pose.orientation.w = quat.w();

        point_cov.scale.x = p.intensity * visual_ray_scale;
        point_cov.scale.y = p.intensity * visual_ray_scale;
        point_cov.scale.z = p.intensity * visual_ray_scale;

        point_cov.color.r = 1.0;
        point_cov.color.g = 0.0;
        point_cov.color.b = 0.0;
        point_cov.color.a = visual_a_scale;

        if (point_cov.scale.x < 0.05)
        {
            point_cov.scale.x = 0.05;
            point_cov.scale.y = 0.05;
            point_cov.scale.z = 0.05;
            point_cov.color.r = 0.0;
            point_cov.color.g = 1.0;
            point_cov.color.b = 0.0;
        }
        point_cov.lifetime = ros::Duration();
        cloud_sphere.markers.push_back(point_cov);
    }
    publisher.publish(cloud_sphere);
}

#endif