//
// Created by lewis on 8/27/22.
//

#include <utility>

#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "cont2_ros/spinner_ros.h"

const std::string PROJ_DIR = std::string(PJSRCDIR);

class BatchBinSpinner : public BaseROSSpinner {
  // --- Added members for evaluation and LC module running ---
  ContourDB contour_db;

  ContLCDEvaluator evaluator;

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds

  // bookkeeping
  int cnt_tp = 0, cnt_fn = 0, cnt_fp = 0;
  double ts_beg = -1;

public:
  explicit BatchBinSpinner(ros::NodeHandle &nh_, const ContourDBConfig &db_config, std::vector<int> q_levels,
                           const std::string &fpath_pose,
                           const std::string &fpath_laser) : BaseROSSpinner(nh_),
                                                             contour_db(db_config, std::move(q_levels)),
                                                             evaluator(fpath_pose, fpath_laser) {

  }

  // before start: 1/1: load thres
  void loadThres(const std::string &thres_fpath) {
    ContLCDEvaluator::loadCheckThres(thres_fpath, thres_lb_, thres_ub_);
  }

  ///
  /// \param outer_cnt
  /// \return if spin successful
  bool spinOnce(int &outer_cnt) {
    bool loaded = evaluator.loadNewScan();
    if (!loaded) {
      printf("Load new scan failed.\n");
      return false;
    }
    TicToc clk;
    ros::Time wall_time_ros = ros::Time::now();
    outer_cnt++;

    // 1. Init current scan
    ContourManagerConfig cm_config;
    cm_config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};

    std::shared_ptr<ContourManager> ptr_cm_tgt = evaluator.getCurrContourManager(cm_config);
    const auto laser_info_tgt = evaluator.getCurrScanInfo();
    printf("\n===\nLoaded: assigned seq: %d, bin path: %s\n", laser_info_tgt.seq, laser_info_tgt.fpath.c_str());

    // 1.1 Prepare and display info: gt/shifted pose, tf
    double ts_curr = laser_info_tgt.ts;
    if (ts_beg < 0) ts_beg = ts_curr;

    Eigen::Isometry3d T_gt_curr = laser_info_tgt.sens_pose;
    Eigen::Vector3d time_translate(0, 0, 1);
    time_translate = time_translate * (ts_curr - ts_beg) / 60;  // 1m per min
    g_poses.insert(std::make_pair(laser_info_tgt.seq, GlobalPoseInfo(T_gt_curr, time_translate.z())));

    geometry_msgs::TransformStamped tf_gt_curr = tf2::eigenToTransform(T_gt_curr);
    broadcastCurrPose(tf_gt_curr);  // the stamp is now

    tf_gt_curr.transform.translation.z += time_translate.z();
    publishPath(wall_time_ros, tf_gt_curr);
    publishScanSeqText(wall_time_ros, tf_gt_curr, laser_info_tgt.seq);


    // 1.2. save images of layers
    clk.tic();
    for (int i = 0; i < cm_config.lv_grads_.size(); i++) {
      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                           ptr_cm_tgt->getStrID() + ".png";   // TODO: what should be the str name of scans?
      ptr_cm_tgt->saveContourImage(f_name, i);
    }
    std::cout << "Time save layers: " << clk.toctic() << std::endl;

    ptr_cm_tgt->clearImage();  // a must to save memory

    // 2. query
    std::vector<std::pair<int, int>> new_lc_pairs;
    std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;

    clk.tic();
    contour_db.queryRangedKNN(ptr_cm_tgt, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs);
    printf("%lu Candidates in %7.5fs: \n", ptr_cands.size(), clk.toc());

    // 2.1 process query results
    CHECK(ptr_cands.size() < 2);
    PredictionOutcome pred_res;
    if (ptr_cands.empty())
      pred_res = evaluator.addPrediction(ptr_cm_tgt);
    else {
      pred_res = evaluator.addPrediction(ptr_cm_tgt, ptr_cands[0], bev_tfs[0]);
      new_lc_pairs.emplace_back(ptr_cm_tgt->getIntID(), ptr_cands[0]->getIntID());
      // save images of pairs
      std::string f_name =
          PROJ_DIR + "/results/match_comp_img/lc_" + ptr_cm_tgt->getStrID() + "-" + ptr_cands[0]->getStrID() +
          ".png";
      ContourManager::saveMatchedPairImg(f_name, *ptr_cm_tgt, *ptr_cands[0]);
      printf("Image saved: %s-%s\n", ptr_cm_tgt->getStrID().c_str(), ptr_cands[0]->getStrID().c_str());

    }

    switch (pred_res.tfpn) {
      case PredictionOutcome::TP:
        printf("Prediction outcome: TP\n");
        cnt_tp++;
        break;
      case PredictionOutcome::FP:
        printf("Prediction outcome: FP\n");
        cnt_fp++;
        break;
      case PredictionOutcome::TN:
        printf("Prediction outcome: TN\n");
        break;
      case PredictionOutcome::FN:
        printf("Prediction outcome: FN\n");
        cnt_fn++;
        break;
    }

    printf("TP Error mean: t:%7.4f m, r:%7.4f deg\n", evaluator.getTPMeanTrans(), evaluator.getTPMeanRot());
    printf("TP Error rmse: t:%7.4f m, r:%7.4f deg\n", evaluator.getTPRMSETrans(), evaluator.getTPRMSERot());
    printf("Accumulated tp poses: %d\n", cnt_tp);
    printf("Accumulated fn poses: %d\n", cnt_fn);
    printf("Accumulated fp poses: %d\n", cnt_fp);

    // 3. update database
    // add scan
    contour_db.addScan(ptr_cm_tgt, laser_info_tgt.ts);
    // balance
    clk.tic();
    contour_db.pushAndBalance(laser_info_tgt.seq, laser_info_tgt.ts);
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

    // 4. publish new vis
    publishLCConnections(new_lc_pairs, wall_time_ros);

    return true;
  }

  void savePredictionResults(const std::string &sav_path) const {
    evaluator.savePredictionResults(sav_path);
  }
};


int main(int argc, char **argv) {
  ros::init(argc, argv, "batch_bin_test");
  ros::NodeHandle nh;

  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  printf("batch bin test start\n");

  const int max_spin_cnt = 123456;

  // Two file path
  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti00.txt";
  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti00.txt";
  // Check thres path
  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti00.txt";

  // Main process:
  ContourDBConfig db_config;
  std::vector<int> db_q_levels = {1, 2, 3};

  BatchBinSpinner o(nh, db_config, db_q_levels, fpath_sens_gt_pose, fpath_lidar_bins);
  o.loadThres(cand_score_config);

  ros::Rate rate(30);
  int cnt = 0;

  while (ros::ok()) {
    ros::spinOnce();
    if (cnt > max_spin_cnt) {
      printf("Max valid spin times %d reached. Exiting the loop...\n", max_spin_cnt);
      break;
    }

    bool ret = o.spinOnce(cnt);
    if (!ret)
      ros::Duration(1.0).sleep();

    rate.sleep();
  }

  o.savePredictionResults(fpath_outcome_sav);


  return 0;
}