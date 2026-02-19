#include <plan_manage/planner_manager.h>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "visualization_msgs/Marker.h"
#include "optimizer/poly_traj_utils.hpp"

namespace ego_planner
{

  EGOPlannerManager::EGOPlannerManager() {}
  EGOPlannerManager::~EGOPlannerManager() { std::cout << "des manager" << std::endl; }

  void EGOPlannerManager::initPlanModules(ros::NodeHandle &nh, PlanningVisualization::Ptr vis)
  {
    nh.param("manager/max_vel", pp_.max_vel_, -1.0);
    nh.param("manager/max_acc", pp_.max_acc_, -1.0);
    nh.param("manager/feasibility_tolerance", pp_.feasibility_tolerance_, 0.0);
    nh.param("manager/polyTraj_piece_length", pp_.polyTraj_piece_length, -1.0);
    nh.param("manager/planning_horizon", pp_.planning_horizen_, 5.0);
    nh.param("manager/use_multitopology_trajs", pp_.use_multitopology_trajs, false);
    nh.param("manager/drone_id", pp_.drone_id, -1);

    grid_map_.reset(new GridMap);
    grid_map_->initMap(nh);

    ploy_traj_opt_.reset(new PolyTrajOptimizer);
    ploy_traj_opt_->setParam(nh);
    ploy_traj_opt_->setEnvironment(grid_map_);

    visualization_ = vis;

    ploy_traj_opt_->setSwarmTrajs(&traj_.swarm_traj);
    ploy_traj_opt_->setDroneId(pp_.drone_id);
  }

  // Helper: generate trajectory from headState, tailState, innerPts, durations
  static PPoly3D generateSplineTraj(
      const Eigen::Matrix<double, 3, 3> &headState,
      const Eigen::Matrix<double, 3, 3> &tailState,
      const Eigen::MatrixXd &innerPts,
      const Eigen::VectorXd &durations)
  {
    int piece_num = durations.size();
    WaypointsMat waypoints(innerPts.cols() + 2, 3);
    waypoints.row(0) = headState.col(0).transpose();
    for (int i = 0; i < innerPts.cols(); ++i)
      waypoints.row(i + 1) = innerPts.col(i).transpose();
    waypoints.row(innerPts.cols() + 1) = tailState.col(0).transpose();

    BCs bc;
    bc.start_velocity = headState.col(1);
    bc.start_acceleration = headState.col(2);
    bc.end_velocity = tailState.col(1);
    bc.end_acceleration = tailState.col(2);

    std::vector<double> time_segs(piece_num);
    for (int i = 0; i < piece_num; ++i)
      time_segs[i] = durations(i);

    SplineTrajectory::QuinticSplineND<3> spline;
    spline.update(time_segs, waypoints, 0.0, bc);
    return spline.getTrajectoryCopy();
  }

  // Helper: get durations from PPoly3D
  static Eigen::VectorXd getDurationsFromTraj(const PPoly3D &traj)
  {
    int num_segs = traj.getNumSegments();
    Eigen::VectorXd durs(num_segs);
    for (int i = 0; i < num_segs; ++i)
      durs(i) = (*(traj.begin() + i)).duration();
    return durs;
  }

  // Helper: get max velocity rate from PPoly3D
  static double getMaxVelRate(const PPoly3D &traj)
  {
    double maxVel = 0.0;
    double dt = 0.01;
    for (double t = traj.getStartTime(); t <= traj.getEndTime(); t += dt)
    {
      double vel = traj.evaluate(t, SplineTrajectory::Deriv::Vel).norm();
      if (vel > maxVel)
        maxVel = vel;
    }
    return maxVel;
  }

  static void appendCompareLog(const std::string &line)
  {
    std::string path;
    ros::param::param("debug/compare_log_path", path, std::string("/tmp/ego_compare.log"));
    std::ofstream ofs(path, std::ios::app);
    if (ofs.is_open())
      ofs << line << std::endl;
  }

  // Helper: compare a spline trajectory with MinJerkOpt using the same waypoints/durations
  static void compareTrajWithMinJerk(const PPoly3D &traj,
                                     const Eigen::VectorXd &durations,
                                     int K,
                                     int drone_id,
                                     const char *tag)
  {
    if (durations.size() <= 0)
      return;

    const int N = traj.getNumSegments();
    if (N != durations.size())
    {
      std::ostringstream ss;
      ss << "[compare_" << tag << "] t=" << ros::Time::now().toSec()
         << " drone=" << drone_id
         << " segment_mismatch traj=" << N << " durs=" << durations.size();
      appendCompareLog(ss.str());
      return;
    }

    Eigen::Matrix<double, 3, 3> headState, tailState;
    double t0 = traj.getStartTime();
    double t1 = traj.getEndTime();
    headState << traj.evaluate(t0, SplineTrajectory::Deriv::Pos),
        traj.evaluate(t0, SplineTrajectory::Deriv::Vel),
        traj.evaluate(t0, SplineTrajectory::Deriv::Acc);
    tailState << traj.evaluate(t1, SplineTrajectory::Deriv::Pos),
        traj.evaluate(t1, SplineTrajectory::Deriv::Vel),
        traj.evaluate(t1, SplineTrajectory::Deriv::Acc);

    Eigen::MatrixXd innerPts(3, std::max(0, N - 1));
    if (N > 1)
    {
      const auto &bp = traj.getBreakpoints();
      for (int i = 0; i < N - 1; ++i)
      {
        innerPts.col(i) = traj.evaluate(bp[i + 1], SplineTrajectory::Deriv::Pos);
      }
    }

    poly_traj::MinJerkOpt mjo;
    mjo.reset(headState, tailState, N);
    mjo.generate(innerPts, durations);

    Eigen::MatrixXd cps_minco = mjo.getInitConstraintPoints(K);

    // Sample spline constraint points
    Eigen::MatrixXd cps_spline(3, N * K + 1);
    int idx = 0;
    double t_accum = traj.getStartTime();
    const auto &bp = traj.getBreakpoints();
    for (int i = 0; i < N; ++i)
    {
      double dur = bp[i + 1] - bp[i];
      double step = dur / K;
      for (int j = 0; j <= K; ++j)
      {
        double t = t_accum + step * j;
        cps_spline.col(idx) = traj.evaluate(t, SplineTrajectory::Deriv::Pos);
        if (j != K || (j == K && i == N - 1))
          ++idx;
      }
      t_accum += dur;
    }

    if (cps_minco.cols() != cps_spline.cols())
    {
      std::ostringstream ss;
      ss << "[compare_" << tag << "] t=" << ros::Time::now().toSec()
         << " drone=" << drone_id
         << " cps_mismatch minco=" << cps_minco.cols()
         << " spline=" << cps_spline.cols();
      appendCompareLog(ss.str());
      return;
    }

    double max_dev = 0.0;
    for (int i = 0; i < cps_minco.cols(); ++i)
    {
      double d = (cps_minco.col(i) - cps_spline.col(i)).norm();
      if (d > max_dev)
        max_dev = d;
    }
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << "[compare_" << tag << "] t=" << ros::Time::now().toSec()
       << " drone=" << drone_id
       << " max_dev=" << std::setprecision(6) << max_dev
       << " cols=" << cps_minco.cols();
    appendCompareLog(ss.str());
  }

  bool EGOPlannerManager::reboundReplan(
      const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_vel,
      const Eigen::Vector3d &start_acc, const Eigen::Vector3d &local_target_pt,
      const Eigen::Vector3d &local_target_vel, const bool flag_polyInit,
      const bool flag_randomPolyTraj, const bool touch_goal)
  {
    ros::Time t_start = ros::Time::now();
    ros::Duration t_init, t_opt;

    static int count = 0;
    std::cout << "\033[47;30m\n[" << t_start << "] Drone " << pp_.drone_id << " Replan " << count++ << "\033[0m" << std::endl;

    /*** STEP 1: INIT ***/
    ploy_traj_opt_->setIfTouchGoal(touch_goal);
    double ts = pp_.polyTraj_piece_length / pp_.max_vel_;

    PPoly3D initTraj;
    Eigen::MatrixXd innerPts;
    Eigen::VectorXd durations;
    Eigen::Matrix<double, 3, 3> headState, tailState;

    if (!computeInitState(start_pt, start_vel, start_acc, local_target_pt, local_target_vel,
                          flag_polyInit, flag_randomPolyTraj, ts,
                          initTraj, innerPts, durations, headState, tailState))
    {
      return false;
    }

    // Optional debug: compare init trajectory with MinJerkOpt for the same inputs
    {
      bool debug_compare_init = false;
      ros::param::param("debug/compare_init_traj", debug_compare_init, false);
      if (debug_compare_init)
      {
        Eigen::Vector3d dir = local_target_pt - start_pt;
        double dir_norm = dir.norm();
        if (dir_norm > 1e-6)
        {
          dir /= dir_norm;
          double sv_par = start_vel.dot(dir);
          double lv_par = local_target_vel.dot(dir);
          double sv_norm = start_vel.norm();
          double lv_norm = local_target_vel.norm();
          double sv_perp = std::sqrt(std::max(0.0, sv_norm * sv_norm - sv_par * sv_par));
          double lv_perp = std::sqrt(std::max(0.0, lv_norm * lv_norm - lv_par * lv_par));

          std::ostringstream ss;
          ss.setf(std::ios::fixed);
          ss << "[compare_init_traj] t=" << ros::Time::now().toSec()
             << " drone=" << pp_.drone_id
             << " dir_norm=" << std::setprecision(3) << dir_norm
             << " sv_par=" << std::setprecision(3) << sv_par
             << " sv_perp=" << std::setprecision(3) << sv_perp
             << " sv_norm=" << std::setprecision(3) << sv_norm
             << " lv_par=" << std::setprecision(3) << lv_par
             << " lv_perp=" << std::setprecision(3) << lv_perp
             << " lv_norm=" << std::setprecision(3) << lv_norm;
          appendCompareLog(ss.str());
        }

        poly_traj::MinJerkOpt mjo;
        mjo.reset(headState, tailState, durations.size());
        mjo.generate(innerPts, durations);
        Eigen::MatrixXd cps_minco = mjo.getInitConstraintPoints(ploy_traj_opt_->get_cps_num_prePiece_());
        Eigen::MatrixXd cps_spline = ploy_traj_opt_->getInitConstraintPoints(initTraj, durations, ploy_traj_opt_->get_cps_num_prePiece_());
        if (cps_minco.cols() == cps_spline.cols())
        {
          double max_dev = 0.0;
          for (int i = 0; i < cps_minco.cols(); ++i)
          {
            double d = (cps_minco.col(i) - cps_spline.col(i)).norm();
            if (d > max_dev)
              max_dev = d;
          }
          std::ostringstream ss;
          ss.setf(std::ios::fixed);
          ss << "[compare_init_traj] t=" << ros::Time::now().toSec()
             << " drone=" << pp_.drone_id
             << " max_dev=" << std::setprecision(6) << max_dev
             << " cols=" << cps_minco.cols();
          appendCompareLog(ss.str());
        }
        else
        {
          std::ostringstream ss;
          ss << "[compare_init_traj] t=" << ros::Time::now().toSec()
             << " drone=" << pp_.drone_id
             << " size_mismatch minco=" << cps_minco.cols()
             << " spline=" << cps_spline.cols();
          appendCompareLog(ss.str());
        }
      }
    }

    Eigen::MatrixXd cstr_pts = ploy_traj_opt_->getInitConstraintPoints(initTraj, durations, ploy_traj_opt_->get_cps_num_prePiece_());
    std::vector<std::pair<int, int>> segments;
    if (ploy_traj_opt_->finelyCheckAndSetConstraintPoints(segments, initTraj, cstr_pts, true) == PolyTrajOptimizer::CHK_RET::ERR)
    {
      return false;
    }

    t_init = ros::Time::now() - t_start;

    std::vector<Eigen::Vector3d> point_set;
    for (int i = 0; i < cstr_pts.cols(); ++i)
      point_set.push_back(cstr_pts.col(i));
    visualization_->displayInitPathList(point_set, 0.2, 0);

    t_start = ros::Time::now();

    /*** STEP 2: OPTIMIZE ***/
    bool flag_success = false;
    std::vector<std::vector<Eigen::Vector3d>> vis_trajs;

    if (pp_.use_multitopology_trajs)
    {
      std::vector<ConstraintPoints> trajs = ploy_traj_opt_->distinctiveTrajs(segments);
      Eigen::VectorXi success = Eigen::VectorXi::Zero(trajs.size());
      double final_cost, min_cost = 999999.0;
      PPoly3D best_traj;
      Eigen::VectorXd best_durations;

      for (int i = trajs.size() - 1; i >= 0; i--)
      {
        ploy_traj_opt_->setConstraintPoints(trajs[i]);
        ploy_traj_opt_->setUseMultitopologyTrajs(true);
        if (ploy_traj_opt_->optimizeTrajectory(headState, tailState,
                                               innerPts, durations, final_cost))
        {
          success[i] = true;

          if (final_cost < min_cost)
          {
            min_cost = final_cost;
            const SplineTraj *opt_spline = ploy_traj_opt_->getSplineOpt().getOptimalSpline();
            if (opt_spline)
            {
              best_traj = opt_spline->getTrajectoryCopy();
              best_durations = getDurationsFromTraj(best_traj);
            }
            flag_success = true;
          }

          // visualization
          const SplineTraj *vis_spline = ploy_traj_opt_->getSplineOpt().getOptimalSpline();
          if (vis_spline)
          {
            PPoly3D vis_traj = vis_spline->getTrajectoryCopy();
            Eigen::VectorXd vis_durs = getDurationsFromTraj(vis_traj);
            Eigen::MatrixXd ctrl_pts_temp = ploy_traj_opt_->getInitConstraintPoints(vis_traj, vis_durs, ploy_traj_opt_->get_cps_num_prePiece_());
            std::vector<Eigen::Vector3d> vis_pts;
            for (int j = 0; j < ctrl_pts_temp.cols(); j++)
              vis_pts.push_back(ctrl_pts_temp.col(j));
            vis_trajs.push_back(vis_pts);
          }
        }
      }

      t_opt = ros::Time::now() - t_start;

      if (trajs.size() > 1)
      {
        std::cout << "\033[1;33m" << "multi-trajs=" << trajs.size() << ",\033[1;0m"
                  << " Success:fail=" << success.sum() << ":" << success.size() - success.sum() << std::endl;
      }

      visualization_->displayMultiOptimalPathList(vis_trajs, 0.1);

      if (flag_success)
      {
        setLocalTrajFromOpt(best_traj, best_durations, touch_goal);
        cstr_pts = ploy_traj_opt_->getInitConstraintPoints(best_traj, best_durations, ploy_traj_opt_->get_cps_num_prePiece_());
        visualization_->displayOptimalList(cstr_pts, 0);

        bool debug_compare_opt = false;
        ros::param::param("debug/compare_opt_traj", debug_compare_opt, false);
        if (debug_compare_opt)
        {
          compareTrajWithMinJerk(best_traj, best_durations, ploy_traj_opt_->get_cps_num_prePiece_(), pp_.drone_id, "opt_traj");
        }
      }
    }
    else
    {
      double final_cost;
      flag_success = ploy_traj_opt_->optimizeTrajectory(headState, tailState,
                                                        innerPts, durations, final_cost);

      t_opt = ros::Time::now() - t_start;

      if (flag_success)
      {
        const SplineTraj *opt_spline = ploy_traj_opt_->getSplineOpt().getOptimalSpline();
        if (opt_spline)
        {
          PPoly3D opt_traj = opt_spline->getTrajectoryCopy();
          Eigen::VectorXd opt_durs = getDurationsFromTraj(opt_traj);
          setLocalTrajFromOpt(opt_traj, opt_durs, touch_goal);
          cstr_pts = ploy_traj_opt_->getInitConstraintPoints(opt_traj, opt_durs, ploy_traj_opt_->get_cps_num_prePiece_());
          visualization_->displayOptimalList(cstr_pts, 0);

          bool debug_compare_opt = false;
          ros::param::param("debug/compare_opt_traj", debug_compare_opt, false);
          if (debug_compare_opt)
          {
            compareTrajWithMinJerk(opt_traj, opt_durs, ploy_traj_opt_->get_cps_num_prePiece_(), pp_.drone_id, "opt_traj");
          }
        }
      }
    }

    /*** STEP 3: Store and display results ***/
    std::cout << "Success=" << (flag_success ? "yes" : "no") << std::endl;
    if (flag_success)
    {
      static double sum_time = 0;
      static int count_success = 0;
      sum_time += (t_init + t_opt).toSec();
      count_success++;
      printf("Time:\033[42m%.3fms,\033[0m init:%.3fms, optimize:%.3fms, avg=%.3fms\n",
             (t_init + t_opt).toSec() * 1000, t_init.toSec() * 1000, t_opt.toSec() * 1000, sum_time / count_success * 1000);

      continous_failures_count_ = 0;
    }
    else
    {
      const SplineTraj *fail_spline = ploy_traj_opt_->getSplineOpt().getOptimalSpline();
      if (fail_spline)
      {
        PPoly3D fail_traj = fail_spline->getTrajectoryCopy();
        Eigen::VectorXd fail_durs = getDurationsFromTraj(fail_traj);
        cstr_pts = ploy_traj_opt_->getInitConstraintPoints(fail_traj, fail_durs, ploy_traj_opt_->get_cps_num_prePiece_());
        visualization_->displayFailedList(cstr_pts, 0);
      }

      continous_failures_count_++;
    }

    return flag_success;
  }

  bool EGOPlannerManager::computeInitState(
      const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
      const Eigen::Vector3d &local_target_pt, const Eigen::Vector3d &local_target_vel,
      const bool flag_polyInit, const bool flag_randomPolyTraj, const double &ts,
      PPoly3D &initTraj, Eigen::MatrixXd &outInnerPts, Eigen::VectorXd &outDurations,
      Eigen::Matrix<double, 3, 3> &headState, Eigen::Matrix<double, 3, 3> &tailState)
  {
    static bool flag_first_call = true;

    auto apply_boundary_debug = [&](Eigen::Matrix<double, 3, 3> &head,
                                    Eigen::Matrix<double, 3, 3> &tail)
    {
      bool force_zero = false;
      bool force_project = false;
      bool force_cap = false;
      ros::param::param("debug/force_zero_boundary_vel", force_zero, false);
      ros::param::param("debug/force_project_boundary_vel", force_project, false);
      ros::param::param("debug/force_cap_boundary_vel", force_cap, false);
      if (!force_zero && !force_project && !force_cap)
        return;

      Eigen::Vector3d dir = local_target_pt - start_pt;
      double dn = dir.norm();
      if (dn < 1e-6)
        return;
      dir /= dn;

      if (force_zero)
      {
        head.col(1).setZero();
        tail.col(1).setZero();
      }
      if (force_project)
      {
        head.col(1) = dir * head.col(1).dot(dir);
        tail.col(1) = dir * tail.col(1).dot(dir);
      }
      if (force_cap)
      {
        double max_v = pp_.max_vel_;
        double hv = head.col(1).norm();
        double tv = tail.col(1).norm();
        if (max_v > 0)
        {
          if (hv > max_v && hv > 1e-6)
            head.col(1) *= (max_v / hv);
          if (tv > max_v && tv > 1e-6)
            tail.col(1) *= (max_v / tv);
        }
      }

      std::ostringstream ss;
      ss << "[compare_init_traj] t=" << ros::Time::now().toSec()
         << " drone=" << pp_.drone_id
         << " boundary_override"
         << " zero=" << (force_zero ? 1 : 0)
         << " proj=" << (force_project ? 1 : 0)
         << " cap=" << (force_cap ? 1 : 0);
      appendCompareLog(ss.str());
    };

    auto log_init_summary = [&](const char *tag,
                                const Eigen::Matrix<double, 3, 3> &head,
                                const Eigen::Matrix<double, 3, 3> &tail,
                                double dist, int piece_nums, double ts_val)
    {
      bool enable = false;
      ros::param::param("debug/log_init_summary", enable, false);
      if (!enable)
        return;

      std::ostringstream ss;
      ss.setf(std::ios::fixed);
      ss << "[init_summary] t=" << ros::Time::now().toSec()
         << " drone=" << pp_.drone_id
         << " tag=" << tag
         << " dist=" << std::setprecision(3) << dist
         << " piece_nums=" << piece_nums
         << " ts=" << std::setprecision(3) << ts_val
         << " start_p=(" << head(0, 0) << "," << head(1, 0) << "," << head(2, 0) << ")"
         << " target_p=(" << tail(0, 0) << "," << tail(1, 0) << "," << tail(2, 0) << ")"
         << " start_v_norm=" << std::setprecision(3) << head.col(1).norm()
         << " target_v_norm=" << std::setprecision(3) << tail.col(1).norm()
         << " start_a_norm=" << std::setprecision(3) << head.col(2).norm();
      appendCompareLog(ss.str());
    };

    if (flag_first_call || flag_polyInit)
    {
      flag_first_call = false;

      Eigen::MatrixXd innerPs;
      Eigen::VectorXd piece_dur_vec;
      int piece_nums;
      constexpr double init_of_init_totaldur = 2.0;
      headState << start_pt, start_vel, start_acc;
      tailState << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();
      apply_boundary_debug(headState, tailState);

      if (!flag_randomPolyTraj)
      {
        piece_nums = 1;
        piece_dur_vec.resize(1);
        piece_dur_vec(0) = init_of_init_totaldur;
      }
      else
      {
        Eigen::Vector3d horizen_dir = ((start_pt - local_target_pt).cross(Eigen::Vector3d(0, 0, 1))).normalized();
        Eigen::Vector3d vertical_dir = ((start_pt - local_target_pt).cross(horizen_dir)).normalized();
        innerPs.resize(3, 1);
        innerPs = (start_pt + local_target_pt) / 2 +
                  (((double)rand()) / RAND_MAX - 0.5) *
                      (start_pt - local_target_pt).norm() *
                      horizen_dir * 0.8 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989) +
                  (((double)rand()) / RAND_MAX - 0.5) *
                      (start_pt - local_target_pt).norm() *
                      vertical_dir * 0.4 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989);

        piece_nums = 2;
        piece_dur_vec.resize(2);
        piece_dur_vec = Eigen::Vector2d(init_of_init_totaldur / 2, init_of_init_totaldur / 2);
      }

      // Generate init of init trajectory
      PPoly3D initOfInitTraj = generateSplineTraj(headState, tailState, innerPs, piece_dur_vec);

      // Generate the real init trajectory
      double dist = (headState.col(0) - tailState.col(0)).norm();
      piece_nums = round(dist / pp_.polyTraj_piece_length);
      if (piece_nums < 2)
        piece_nums = 2;
      double piece_dur = init_of_init_totaldur / (double)piece_nums;
      piece_dur_vec.resize(piece_nums);
      piece_dur_vec = Eigen::VectorXd::Constant(piece_nums, ts);
      log_init_summary("poly_init", headState, tailState, dist, piece_nums, ts);
      innerPs.resize(3, piece_nums - 1);
      int id = 0;
      double t_s = piece_dur, t_e = init_of_init_totaldur - piece_dur / 2;
      double start_time = initOfInitTraj.getStartTime();
      for (double t = t_s; t < t_e; t += piece_dur)
      {
        innerPs.col(id++) = initOfInitTraj.evaluate(start_time + t, SplineTrajectory::Deriv::Pos);
      }
      if (id != piece_nums - 1)
      {
        ROS_ERROR("Should not happen! x_x");
        return false;
      }

      initTraj = generateSplineTraj(headState, tailState, innerPs, piece_dur_vec);

      outInnerPts = innerPs;
      outDurations = piece_dur_vec;
    }
    else
    {
      if (traj_.global_traj.last_glb_t_of_lc_tgt < 0.0)
      {
        ROS_ERROR("You are initialzing a trajectory from a previous optimal trajectory, but no previous trajectories up to now.");
        return false;
      }

      double passed_t_on_lctraj = ros::Time::now().toSec() - traj_.local_traj.start_time;
      double t_to_lc_end = traj_.local_traj.duration - passed_t_on_lctraj;
      if (t_to_lc_end < 0)
      {
        ROS_INFO("t_to_lc_end < 0, exit and wait for another call.");
        return false;
      }
      double t_to_lc_tgt = t_to_lc_end +
                           (traj_.global_traj.glb_t_of_lc_tgt - traj_.global_traj.last_glb_t_of_lc_tgt);
      double dist = (start_pt - local_target_pt).norm();
      int piece_nums = ceil(dist / pp_.polyTraj_piece_length);
      if (piece_nums < 2)
        piece_nums = 2;

      headState << start_pt, start_vel, start_acc;
      tailState << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();
      apply_boundary_debug(headState, tailState);

      Eigen::MatrixXd innerPs(3, piece_nums - 1);
      Eigen::VectorXd piece_dur_vec = Eigen::VectorXd::Constant(piece_nums, t_to_lc_tgt / piece_nums);
      log_init_summary("prev_opt", headState, tailState, dist, piece_nums, t_to_lc_tgt / piece_nums);

      double t = piece_dur_vec(0);
      double lc_start = traj_.local_traj.traj.getStartTime();
      double glb_start = traj_.global_traj.traj.getStartTime();
      for (int i = 0; i < piece_nums - 1; ++i)
      {
        if (t < t_to_lc_end)
        {
          innerPs.col(i) = traj_.local_traj.traj.evaluate(lc_start + t + passed_t_on_lctraj, SplineTrajectory::Deriv::Pos);
        }
        else if (t <= t_to_lc_tgt)
        {
          double glb_t = t - t_to_lc_end + traj_.global_traj.last_glb_t_of_lc_tgt - traj_.global_traj.global_start_time;
          innerPs.col(i) = traj_.global_traj.traj.evaluate(glb_start + glb_t, SplineTrajectory::Deriv::Pos);
        }
        else
        {
          ROS_ERROR("Should not happen! x_x 0x88 t=%.2f, t_to_lc_end=%.2f, t_to_lc_tgt=%.2f", t, t_to_lc_end, t_to_lc_tgt);
        }

        t += piece_dur_vec(i + 1);
      }

      initTraj = generateSplineTraj(headState, tailState, innerPs, piece_dur_vec);
      outInnerPts = innerPs;
      outDurations = piece_dur_vec;
    }

    return true;
  }

  void EGOPlannerManager::getLocalTarget(
      const double planning_horizen, const Eigen::Vector3d &start_pt,
      const Eigen::Vector3d &global_end_pt, Eigen::Vector3d &local_target_pos,
      Eigen::Vector3d &local_target_vel, bool &touch_goal)
  {
    double t;
    touch_goal = false;

    traj_.global_traj.last_glb_t_of_lc_tgt = traj_.global_traj.glb_t_of_lc_tgt;

    double t_step = planning_horizen / 20 / pp_.max_vel_;
    double glb_start = traj_.global_traj.traj.getStartTime();

    for (t = traj_.global_traj.glb_t_of_lc_tgt;
         t < (traj_.global_traj.global_start_time + traj_.global_traj.duration);
         t += t_step)
    {
      double local_t = t - traj_.global_traj.global_start_time;
      Eigen::Vector3d pos_t = traj_.global_traj.traj.evaluate(glb_start + local_t, SplineTrajectory::Deriv::Pos);
      double dist = (pos_t - start_pt).norm();

      if (dist >= planning_horizen)
      {
        local_target_pos = pos_t;
        traj_.global_traj.glb_t_of_lc_tgt = t;
        break;
      }
    }

    if ((t - traj_.global_traj.global_start_time) >= traj_.global_traj.duration - 1e-5)
    {
      local_target_pos = global_end_pt;
      traj_.global_traj.glb_t_of_lc_tgt = traj_.global_traj.global_start_time + traj_.global_traj.duration;
      touch_goal = true;
    }

    if ((global_end_pt - local_target_pos).norm() < (pp_.max_vel_ * pp_.max_vel_) / (2 * pp_.max_acc_))
    {
      local_target_vel = Eigen::Vector3d::Zero();
    }
    else
    {
      double local_t = t - traj_.global_traj.global_start_time;
      local_target_vel = traj_.global_traj.traj.evaluate(glb_start + local_t, SplineTrajectory::Deriv::Vel);
    }
  }

  bool EGOPlannerManager::setLocalTrajFromOpt(const PPoly3D &traj, const Eigen::VectorXd &durations, const bool touch_goal)
  {
    Eigen::MatrixXd cps = ploy_traj_opt_->getInitConstraintPoints(traj, durations, getCpsNumPrePiece());
    PtsChk_t pts_to_check;
    bool ret = ploy_traj_opt_->computePointsToCheck(traj, ConstraintPoints::two_thirds_id(cps, touch_goal), pts_to_check);
    if (ret && pts_to_check.size() >= 1 && pts_to_check.back().size() >= 1)
    {
      traj_.setLocalTraj(traj, pts_to_check, ros::Time::now().toSec());
    }

    return ret;
  }

  bool EGOPlannerManager::EmergencyStop(Eigen::Vector3d stop_pos)
  {
    auto ZERO = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 3> headState, tailState;
    headState << stop_pos, ZERO, ZERO;
    tailState = headState;
    Eigen::MatrixXd innerPs = stop_pos; // 3x1
    Eigen::VectorXd durs = Eigen::Vector2d(1.0, 1.0);

    PPoly3D stopTraj = generateSplineTraj(headState, tailState, innerPs, durs);
    setLocalTrajFromOpt(stopTraj, durs, false);

    return true;
  }

  bool EGOPlannerManager::checkCollision(int drone_id)
  {
    if (traj_.local_traj.start_time < 1e9)
      return false;
    if (traj_.swarm_traj[drone_id].drone_id != drone_id)
      return false;

    double my_traj_start_time = traj_.local_traj.start_time;
    double other_traj_start_time = traj_.swarm_traj[drone_id].start_time;

    double t_start = std::max(my_traj_start_time, other_traj_start_time);
    double t_end = std::min(my_traj_start_time + traj_.local_traj.duration * 2 / 3,
                            other_traj_start_time + traj_.swarm_traj[drone_id].duration);

    double my_base = traj_.local_traj.traj.getStartTime();
    double other_base = traj_.swarm_traj[drone_id].traj.getStartTime();

    for (double t = t_start; t < t_end; t += 0.03)
    {
      if ((traj_.local_traj.traj.evaluate(my_base + (t - my_traj_start_time), SplineTrajectory::Deriv::Pos) -
           traj_.swarm_traj[drone_id].traj.evaluate(other_base + (t - other_traj_start_time), SplineTrajectory::Deriv::Pos))
              .norm() < (getSwarmClearance() + traj_.swarm_traj[drone_id].des_clearance))
      {
        return true;
      }
    }

    return false;
  }

  bool EGOPlannerManager::planGlobalTrajWaypoints(
      const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel,
      const Eigen::Vector3d &start_acc, const std::vector<Eigen::Vector3d> &waypoints,
      const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {
    Eigen::Matrix<double, 3, 3> headState, tailState;
    headState << start_pos, start_vel, start_acc;
    tailState << waypoints.back(), end_vel, end_acc;
    Eigen::MatrixXd innerPts;

    if (waypoints.size() > 1)
    {
      innerPts.resize(3, waypoints.size() - 1);
      for (int i = 0; i < (int)waypoints.size() - 1; ++i)
        innerPts.col(i) = waypoints[i];
    }

    double des_vel = pp_.max_vel_ / 1.5;
    Eigen::VectorXd time_vec(waypoints.size());

    PPoly3D globalTraj;
    for (int j = 0; j < 2; ++j)
    {
      for (size_t i = 0; i < waypoints.size(); ++i)
      {
        time_vec(i) = (i == 0) ? (waypoints[0] - start_pos).norm() / des_vel
                                : (waypoints[i] - waypoints[i - 1]).norm() / des_vel;
      }

      globalTraj = generateSplineTraj(headState, tailState, innerPts, time_vec);

      if (getMaxVelRate(globalTraj) < pp_.max_vel_ ||
          start_vel.norm() > pp_.max_vel_ ||
          end_vel.norm() > pp_.max_vel_)
      {
        break;
      }

      if (j == 2)
      {
        ROS_WARN("Global traj MaxVel = %f > set_max_vel", getMaxVelRate(globalTraj));
      }

      des_vel /= 1.5;
    }

    auto time_now = ros::Time::now();
    traj_.setGlobalTraj(globalTraj, time_now.toSec());

    return true;
  }

} // namespace ego_planner
