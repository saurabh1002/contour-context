//
// Created by lewis on 5/12/22.
//

#ifndef CONT2_IO_BIN_H
#define CONT2_IO_BIN_H

/* Read binary data without ROS interface
 * */

#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Read KITTI lidar bin (same for MulRan)
// assumptions/approximations:
//  1. every lidar frame has a corresponding gt pose
//  2. timestamps and the extrinsic are insignificant
class ReadKITTILiDAR {
  const std::string kitti_raw_dir_, date_, seq_;

  std::vector<std::pair<int, Eigen::Isometry3d>> imu_gt_poses_;
  std::vector<std::pair<int, std::string>> lidar_ts_, imu_ts_;

  int max_index_num = 10000;
  int seq_name_len = 10; // ".bin", ".txt" excluded


public:
  explicit ReadKITTILiDAR(std::string &kitti_raw_dir, std::string &date, std::string &seq) : kitti_raw_dir_(
      kitti_raw_dir), date_(date), seq_(seq) {
    //TODO
    double scale = 0;
    Eigen::Vector3d trans_orig(0, 0, 0);
    for (int idx = 0; idx < max_index_num; idx++) {
      std::string idx_str = std::to_string(idx);
      std::string imu_entry_path = kitti_raw_dir_ + "/" + date_ + "/" + seq_ + "/oxts/data/" +
                                   std::string(seq_name_len - idx_str.length(), '0') + idx_str + ".txt";
      std::fstream infile;
      infile.open(imu_entry_path, std::ios::in);
      if (infile.rdstate() != std::ifstream::goodbit) {
        std::cout << "Cannot open " << imu_entry_path << ", breaking loop..." << std::endl;
        break;
      }
      std::string sbuf, pname;
      std::getline(infile, sbuf); // the data has only one line
      std::istringstream iss(sbuf);

      double pose_dat[6];
      for (double &i: pose_dat)
        iss >> i;

      infile.close();

      // https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
      double er = 6378137.0;
      if (scale == 0) {
        scale = std::cos(pose_dat[0] * M_PI / 180.0);
      }

      Eigen::Vector3d trans(scale * pose_dat[1] * M_PI * er / 180,
                            scale * er * std::log(std::tan((90 + pose_dat[0]) * M_PI / 360)),
                            pose_dat[2]);
      Eigen::Quaterniond rot = Eigen::Quaterniond(Eigen::AngleAxisd(pose_dat[5], Eigen::Vector3d::UnitZ())
                                                  * Eigen::AngleAxisd(pose_dat[4], Eigen::Vector3d::UnitY())
                                                  * Eigen::AngleAxisd(pose_dat[3], Eigen::Vector3d::UnitX()));
      if (trans_orig.sum() == 0)
        trans_orig = trans;

      trans = trans - trans_orig;
      Eigen::Isometry3d res;
      res.setIdentity();
      res.rotate(rot);
      res.pretranslate(trans);
      imu_gt_poses_.emplace_back(idx, res);
    }
  };

  // get all gt pose (to display)
  std::vector<std::pair<int, Eigen::Isometry3d>> getGtImuPoses() const {
    return imu_gt_poses_;
  }

  // get point cloud
  // we may not need to display it in rviz
  template<typename PointType>
  typename pcl::PointCloud<PointType>::ConstPtr getLidarPointCloud(int idx, std::string &str_idx0lead) {
    typename pcl::PointCloud<PointType>::Ptr out_ptr = nullptr;

    std::string idx_str = std::to_string(idx);
    str_idx0lead = std::string(seq_name_len - idx_str.length(), '0') + idx_str;
    std::string lidar_bin_path =
        kitti_raw_dir_ + "/" + date_ + "/" + seq_ + "/velodyne_points/data/" + str_idx0lead + ".bin";

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int num = 1000000;
    auto *data = (float *) malloc(num * sizeof(float));
    // pointers
    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pr = data + 3;

    FILE *stream;
    stream = fopen(lidar_bin_path.c_str(), "rb");
    if (stream) {
      num = fread(data, sizeof(float), num, stream) / 4;
      out_ptr.reset(new pcl::PointCloud<PointType>());
      out_ptr->reserve(num);
      for (int32_t i = 0; i < num; i++) {
        PointType pt;
        pt.x = *px;
        pt.y = *py;
        pt.z = *pz;
        out_ptr->push_back(pt);

        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
      }
      fclose(stream);

    } else {
      printf("Lidar bin file %s does not exist.\n", lidar_bin_path.c_str());
    }
    return out_ptr;
  }


};

// Read PointCloud2 data
class ReadPointCloud2 {
public:
};

#endif //CONT2_IO_BIN_H
