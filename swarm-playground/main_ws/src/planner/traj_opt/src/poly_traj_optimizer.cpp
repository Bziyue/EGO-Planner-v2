#include "optimizer/poly_traj_optimizer.h"

using namespace std;

#define VERBOSE_OUTPUT false
#define PRINTF_COND(STR, ...) \
  if (VERBOSE_OUTPUT)         \
  printf(STR, __VA_ARGS__)

namespace ego_planner
{
  // =====================================================
  //  Generate trajectory from states using QuinticSplineND
  // =====================================================
  PPoly3D PolyTrajOptimizer::generateTrajectory(
      const Eigen::MatrixXd &iniState, const Eigen::MatrixXd &finState,
      const Eigen::MatrixXd &innerPts, const Eigen::VectorXd &durations)
  {
    int piece_num = durations.size();
    // Build waypoints: start + inner + end
    WaypointsVec waypoints;
    waypoints.push_back(iniState.col(0)); // start position
    for (int i = 0; i < innerPts.cols(); ++i)
      waypoints.push_back(innerPts.col(i));
    waypoints.push_back(finState.col(0)); // end position

    // Build boundary conditions
    BCs bc;
    bc.start_velocity = iniState.col(1);
    bc.start_acceleration = iniState.col(2);
    bc.end_velocity = finState.col(1);
    bc.end_acceleration = finState.col(2);

    // Build time segments
    std::vector<double> time_segs(piece_num);
    for (int i = 0; i < piece_num; ++i)
      time_segs[i] = durations(i);

    // Create and update spline
    SplineTraj spline;
    spline.update(time_segs, waypoints, 0.0, bc);

    return spline.getTrajectoryCopy();
  }

  // =====================================================
  //  Get initial constraint points from a PPoly3D trajectory
  // =====================================================
  Eigen::MatrixXd PolyTrajOptimizer::getInitConstraintPoints(
      const PPoly3D &traj,
      const Eigen::VectorXd &durations,
      int K) const
  {
    int N = durations.size();
    int total_pts = N * K + 1;
    Eigen::MatrixXd cstr_pts(3, total_pts);
    int idx = 0;
    double t_accum = traj.getStartTime();

    ROS_INFO("[DEBUG] getInitConstraintPoints: N=%d, K=%d, total_pts=%d, startTime=%.4f, numSegs=%d",
             N, K, total_pts, traj.getStartTime(), traj.getNumSegments());

    for (int i = 0; i < N; ++i)
    {
      double dur = durations(i);
      double step = dur / K;
      for (int j = 0; j <= K; ++j)
      {
        double t = t_accum + step * j;
        cstr_pts.col(idx) = traj.evaluate(t, SplineTrajectory::Deriv::Pos);
        if (i == 0 && j <= 1) // print first two points
        {
          ROS_INFO("[DEBUG] cstr_pts[%d] t=%.4f pos=(%.3f,%.3f,%.3f)",
                   idx, t, cstr_pts(0,idx), cstr_pts(1,idx), cstr_pts(2,idx));
        }
        if (j != K || (j == K && i == N - 1))
          ++idx;
      }
      t_accum += dur;
    }

    // Print last point
    ROS_INFO("[DEBUG] cstr_pts[%d] (last) pos=(%.3f,%.3f,%.3f)",
             total_pts-1, cstr_pts(0,total_pts-1), cstr_pts(1,total_pts-1), cstr_pts(2,total_pts-1));

    return cstr_pts;
  }

  // =====================================================
  //  Main optimization API
  // =====================================================
  bool PolyTrajOptimizer::optimizeTrajectory(
      const Eigen::MatrixXd &iniState, const Eigen::MatrixXd &finState,
      const Eigen::MatrixXd &initInnerPts, const Eigen::VectorXd &initT,
      double &final_cost)
  {
    if (initInnerPts.cols() != (initT.size() - 1))
    {
      ROS_ERROR("initInnerPts.cols() != (initT.size()-1)");
      return false;
    }

    ros::Time t0 = ros::Time::now(), t1, t2;
    int restart_nums = 0, rebound_times = 0;
    bool flag_force_return, flag_still_unsafe, flag_success, flag_swarm_too_close;
    multitopology_data_.initial_obstacles_avoided = false;
    wei_swarm_mod_ = wei_swarm_;

    t_now_ = ros::Time::now().toSec();
    piece_num_ = initT.size();

    // Setup SplineOptimizer
    WaypointsVec waypoints;
    waypoints.push_back(iniState.col(0));
    for (int i = 0; i < initInnerPts.cols(); ++i)
      waypoints.push_back(initInnerPts.col(i));
    waypoints.push_back(finState.col(0));

    BCs bc;
    bc.start_velocity = iniState.col(1);
    bc.start_acceleration = iniState.col(2);
    bc.end_velocity = finState.col(1);
    bc.end_acceleration = finState.col(2);

    std::vector<double> time_segs(piece_num_);
    for (int i = 0; i < piece_num_; ++i)
      time_segs[i] = initT(i);

    splineOpt_.setInitState(time_segs, waypoints, 0.0, bc);

    // Only optimize inner waypoints and times, fix boundary states
    SplineTrajectory::OptimizationFlags flags;
    flags.start_p = false;
    flags.end_p = false;
    flags.start_v = false;
    flags.end_v = false;
    flags.start_a = false;
    flags.end_a = false;
    splineOpt_.setOptimizationFlags(flags);
    splineOpt_.setEnergyWeights(1.0); // energy (jerk integral) must counterbalance time cost to prevent LBFGS line-search failure
    splineOpt_.setIntegralNumSteps(cps_num_prePiece_);

    // Generate initial guess
    Eigen::VectorXd x0 = splineOpt_.generateInitialGuess();
    variable_num_ = x0.size();

    ROS_INFO("[DEBUG] optimizeTrajectory: piece_num=%d, variable_num=%d, x0_norm=%.6f",
             piece_num_, variable_num_, x0.norm());
    // Print first few x0 values
    {
      std::string x0_str = "[";
      for (int i = 0; i < std::min(variable_num_, 10); ++i)
        x0_str += std::to_string(x0(i)) + (i < std::min(variable_num_, 10) - 1 ? "," : "");
      if (variable_num_ > 10) x0_str += ",...";
      x0_str += "]";
      ROS_INFO("[DEBUG] x0 (first 10): %s", x0_str.c_str());
    }

    // Copy to raw array for LBFGS
    double x_init[variable_num_];
    memcpy(x_init, x0.data(), variable_num_ * sizeof(double));

    min_ellip_dist2_.resize(swarm_trajs_->size());

    // Setup cost functions
    time_cost_func_.wei_time = wei_time_;

    integral_cost_func_.grid_map = grid_map_;
    integral_cost_func_.cps = &cps_;
    integral_cost_func_.swarm_trajs = swarm_trajs_;
    integral_cost_func_.wei_obs = wei_obs_;
    integral_cost_func_.wei_obs_soft = wei_obs_soft_;
    integral_cost_func_.wei_swarm = wei_swarm_mod_;
    integral_cost_func_.wei_feas = wei_feas_;
    integral_cost_func_.wei_sqrvar = wei_sqrvar_;
    integral_cost_func_.obs_clearance = obs_clearance_;
    integral_cost_func_.obs_clearance_soft = obs_clearance_soft_;
    integral_cost_func_.swarm_clearance = swarm_clearance_;
    integral_cost_func_.max_vel = max_vel_;
    integral_cost_func_.max_acc = max_acc_;
    integral_cost_func_.max_jer = max_jer_;
    integral_cost_func_.drone_id = drone_id_;
    integral_cost_func_.t_now = t_now_;
    integral_cost_func_.touch_goal = touch_goal_;
    integral_cost_func_.cps_per_piece = cps_num_prePiece_;
    integral_cost_func_.min_ellip_dist2_ptr = &min_ellip_dist2_;

    // LBFGS params
    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
    lbfgs_params.mem_size = 16;
    lbfgs_params.max_iterations = 200;
    lbfgs_params.min_step = 1e-32;
    lbfgs_params.past = 3;
    lbfgs_params.delta = 1.0e-2;

    do
    {
      iter_num_ = 0;
      flag_force_return = false;
      force_stop_type_ = DONT_STOP;
      flag_still_unsafe = false;
      flag_success = false;
      flag_swarm_too_close = false;

      // Update swarm weight for retry
      integral_cost_func_.wei_swarm = wei_swarm_mod_;

      t1 = ros::Time::now();
      int result = lbfgs::lbfgs_optimize(
          variable_num_,
          x_init,
          &final_cost,
          PolyTrajOptimizer::costFunctionCallback,
          NULL,
          PolyTrajOptimizer::earlyExitCallback,
          this,
          &lbfgs_params);

      t2 = ros::Time::now();
      double time_ms = (t2 - t1).toSec() * 1000;
      double total_time_ms = (t2 - t0).toSec() * 1000;

      ROS_INFO("[DEBUG] LBFGS result=%d (%s), iter=%d, cost=%.6f, time=%.3fms",
               result, lbfgs::lbfgs_strerror(result), iter_num_, final_cost, time_ms);

      if (result == lbfgs::LBFGS_CONVERGENCE ||
          result == lbfgs::LBFGSERR_MAXIMUMITERATION ||
          result == lbfgs::LBFGS_ALREADY_MINIMIZED ||
          result == lbfgs::LBFGS_STOP ||
          result == lbfgs::LBFGSERR_ROUNDING_ERROR)
      {
        flag_force_return = false;

        std::vector<std::pair<int, int>> segments_nouse;
        for (size_t i = 0; i < swarm_trajs_->size(); ++i)
        {
          flag_swarm_too_close |= min_ellip_dist2_[i] < pow((swarm_clearance_ + swarm_trajs_->at(i).des_clearance) * 1.25, 2);
        }
        if (!flag_swarm_too_close)
        {
          // Get optimized trajectory for collision check
          const SplineTraj *opt_spline = splineOpt_.getOptimalSpline();
          ROS_INFO("[DEBUG] getOptimalSpline() returned %s", opt_spline ? "VALID" : "NULL");
          if (opt_spline)
          {
            PPoly3D traj = opt_spline->getTrajectoryCopy();
            Eigen::VectorXd durs(piece_num_);
            for (int i = 0; i < piece_num_; ++i)
              durs(i) = (*(opt_spline->getTrajectory().begin() + i)).duration();

            // Debug: print optimized trajectory info
            {
              double total_dur = durs.sum();
              std::string dur_str = "";
              for (int i = 0; i < piece_num_; ++i)
                dur_str += std::to_string(durs(i)) + (i < piece_num_ - 1 ? "," : "");
              ROS_INFO("[DEBUG] opt_traj: pieces=%d, durations=[%s], total_dur=%.3f",
                       piece_num_, dur_str.c_str(), total_dur);
              // Sample velocity/acceleration at a few points
              double dt_sample = total_dur / 4.0;
              double t0 = traj.getStartTime();
              for (int si = 0; si <= 4; ++si)
              {
                double ts = t0 + dt_sample * si;
                Eigen::Vector3d ps = traj.evaluate(ts, SplineTrajectory::Deriv::Pos);
                Eigen::Vector3d vs = traj.evaluate(ts, SplineTrajectory::Deriv::Vel);
                Eigen::Vector3d as = traj.evaluate(ts, SplineTrajectory::Deriv::Acc);
                ROS_INFO("[DEBUG] opt_traj t=%.3f pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)|%.2f| acc=(%.3f,%.3f,%.3f)|%.2f|",
                         ts, ps.x(), ps.y(), ps.z(),
                         vs.x(), vs.y(), vs.z(), vs.norm(),
                         as.x(), as.y(), as.z(), as.norm());
              }
            }
            Eigen::MatrixXd init_points = getInitConstraintPoints(traj, durs, cps_num_prePiece_);

            if (finelyCheckAndSetConstraintPoints(segments_nouse, traj, init_points, false) == CHK_RET::OBS_FREE)
            {
              flag_success = true;
              PRINTF_COND("\033[32miter=%d,time(ms)=%5.3f,total_t(ms)=%5.3f,cost=%5.3f\n\033[0m", iter_num_, time_ms, total_time_ms, final_cost);
            }
            else
            {
              flag_still_unsafe = true;
              restart_nums++;
              PRINTF_COND("\033[32miter=%d,time(ms)=%5.3f, fine check collided, keep optimizing\n\033[0m", iter_num_, time_ms);
            }
          }
        }
        else
        {
          PRINTF_COND("Swarm clearance not satisfied, keep optimizing. iter=%d,time(ms)=%5.3f, wei_swarm_mod_=%f\n", iter_num_, time_ms, wei_swarm_mod_);
          flag_still_unsafe = true;
          restart_nums++;
          wei_swarm_mod_ *= 2;
        }
      }
      else if (result == lbfgs::LBFGSERR_CANCELED)
      {
        flag_force_return = true;
        rebound_times++;
        PRINTF_COND("iter=%d, time(ms)=%f, rebound\n", iter_num_, time_ms);
      }
      else
      {
        PRINTF_COND("iter=%d, time(ms)=%f, error\n", iter_num_, time_ms);
        ROS_WARN_COND(VERBOSE_OUTPUT, "Solver error. Return = %d, %s. Skip this planning.", result, lbfgs::lbfgs_strerror(result));
      }

    } while ((flag_still_unsafe && restart_nums < 3) ||
             (flag_force_return && force_stop_type_ == STOP_FOR_REBOUND && rebound_times <= 20));

    return flag_success;
  }

  // =====================================================
  //  LBFGS cost function callback
  // =====================================================
  double PolyTrajOptimizer::costFunctionCallback(void *func_data, const double *x, double *grad, const int n)
  {
    PolyTrajOptimizer *opt = reinterpret_cast<PolyTrajOptimizer *>(func_data);

    fill(opt->min_ellip_dist2_.begin(), opt->min_ellip_dist2_.end(), std::numeric_limits<double>::max());

    // Copy from raw pointers to Eigen::VectorXd (Map doesn't bind to VectorXd&)
    Eigen::VectorXd x_vec = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::VectorXd grad_vec = Eigen::VectorXd::Zero(n);

    // Reset integral cost accumulator
    opt->integral_cost_func_.resetAccumulation();

    // Use SplineOptimizer::evaluate to compute cost and gradients
    // Don't pass external workspace — let evaluate use internal_ws_ so getOptimalSpline() works.
    double total_cost = opt->splineOpt_.evaluate(
        x_vec, grad_vec,
        opt->time_cost_func_,
        opt->integral_cost_func_);

    // Copy gradients back to raw pointer
    Eigen::Map<Eigen::VectorXd>(grad, n) = grad_vec;

    // Debug: print cost breakdown every iteration
    {
      // Compute energy and time costs directly for accurate breakdown.
      double energy_raw = 0.0, energy_cost = 0.0, time_cost = 0.0;
      double rho_e = opt->splineOpt_.getEnergyWeight();
      const SplineTraj *cur_spline = opt->splineOpt_.getOptimalSpline();
      if (cur_spline)
      {
        energy_raw = cur_spline->getEnergy();
        energy_cost = rho_e * energy_raw; // actual energy contribution to total_cost
        for (auto &seg_t : cur_spline->getTimeSegments())
          time_cost += seg_t * opt->time_cost_func_.wei_time;
      }
      double weighted_integral = total_cost - energy_cost - time_cost;
      double obs_raw = opt->integral_cost_func_.accumulated_costs(0);
      double swarm_raw = opt->integral_cost_func_.accumulated_costs(1);
      double feas_raw = opt->integral_cost_func_.accumulated_costs(2);
      ROS_INFO("[DEBUG] iter=%d: total=%.1f, integral=%.1f(obs_r=%.1f,swm_r=%.1f,fea_r=%.1f), energy=%.1f(raw=%.1f,rho=%.2f), time=%.1f, grad=%.1f",
               opt->iter_num_, total_cost, weighted_integral, obs_raw, swarm_raw, feas_raw, energy_cost, energy_raw, rho_e, time_cost, grad_vec.norm());
    }

    // Distance variance cost on constraint points (post-processing)
    if (opt->wei_sqrvar_ > 0 && opt->cps_.cp_size > 1)
    {
      Eigen::MatrixXd gdp;
      double var = 0;
      opt->distanceSqrVarianceWithGradCost2p(opt->cps_.points, gdp, var);
      total_cost += var;
      // Note: The variance gradient on constraint points is not easily backpropagated
      // through the spline. For now it contributes to the cost but not gradient.
      // This is acceptable as the variance is a regularization term.
    }

    // Check for rebound
    if (opt->allowRebound())
    {
      opt->roughlyCheckConstraintPoints();
    }

    opt->iter_num_ += 1;
    return total_cost;
  }

  int PolyTrajOptimizer::earlyExitCallback(void *func_data, const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls)
  {
    PolyTrajOptimizer *opt = reinterpret_cast<PolyTrajOptimizer *>(func_data);
    return (opt->force_stop_type_ == STOP_FOR_ERROR || opt->force_stop_type_ == STOP_FOR_REBOUND);
  }

  // =====================================================
  //  Collision checking (adapted for PPoly3D)
  // =====================================================
  bool PolyTrajOptimizer::computePointsToCheck(
      const PPoly3D &traj,
      int id_cps_end, PtsChk_t &pts_check)
  {
    pts_check.clear();
    pts_check.resize(id_cps_end);
    const double RES = grid_map_->getResolution(), RES_2 = RES / 2;

    // Build durations vector from PPoly3D segments
    int num_segs = traj.getNumSegments();
    Eigen::VectorXd durations(num_segs);
    double dur_sum = 0;
    for (int i = 0; i < num_segs; ++i)
    {
      durations(i) = (*(traj.begin() + i)).duration();
      dur_sum += durations(i);
    }

    Eigen::VectorXd t_seg_start(num_segs + 1);
    t_seg_start(0) = 0;
    for (int i = 0; i < num_segs; ++i)
      t_seg_start(i + 1) = t_seg_start(i) + durations(i);

    const double DURATION = dur_sum;
    double t_step = min(RES / max_vel_, durations.minCoeff() / max(cps_num_prePiece_, 1) / 1.5);
    double start_t = traj.getStartTime();
    Eigen::Vector3d pt_last = traj.evaluate(start_t, SplineTrajectory::Deriv::Pos);
    int id_cps_curr = 0, id_piece_curr = 0;

    double t = 0.0;
    while (true)
    {
      if (t > DURATION)
      {
        if (touch_goal_ && pts_check.size() > 0)
        {
          while (pts_check.back().size() == 0)
            pts_check.pop_back();

          if (pts_check.size() <= 0)
          {
            ROS_ERROR("Failed to get points list to check (0x02). pts_check.size()=%d", (int)pts_check.size());
            return false;
          }
          else
            return true;
        }
        else
        {
          ROS_ERROR("Failed to get points list to check (0x01). touch_goal_=%d, pts_check.size()=%d", touch_goal_, (int)pts_check.size());
          pts_check.clear();
          return false;
        }
      }

      const double next_t_stp = t_seg_start(id_piece_curr) + durations(id_piece_curr) / cps_num_prePiece_ * ((id_cps_curr + 1) - cps_num_prePiece_ * id_piece_curr);
      if (t >= next_t_stp)
      {
        if (id_cps_curr + 1 >= cps_num_prePiece_ * (id_piece_curr + 1))
          ++id_piece_curr;
        if (++id_cps_curr >= id_cps_end)
          break;
      }

      Eigen::Vector3d pt = traj.evaluate(start_t + t, SplineTrajectory::Deriv::Pos);
      if (t < 1e-5 || pts_check[id_cps_curr].size() == 0 || (pt - pt_last).cwiseAbs().maxCoeff() > RES_2)
      {
        pts_check[id_cps_curr].emplace_back(std::pair<double, Eigen::Vector3d>(t, pt));
        pt_last = pt;
      }

      t += t_step;
    }

    return true;
  }

  PolyTrajOptimizer::CHK_RET PolyTrajOptimizer::finelyCheckAndSetConstraintPoints(
      std::vector<std::pair<int, int>> &segments,
      const PPoly3D &traj,
      const Eigen::MatrixXd &init_points,
      const bool flag_first_init)
  {
    if (flag_first_init)
    {
      cps_.resize_cp(init_points.cols());
      cps_.points = init_points;
    }

    /*** Segment the initial trajectory according to obstacles ***/
    vector<std::pair<int, int>> segment_ids;
    constexpr int ENOUGH_INTERVAL = 2;
    int in_id = -1, out_id = -1;
    int same_occ_state_times = ENOUGH_INTERVAL + 1;
    bool occ, last_occ = false;
    bool flag_got_start = false, flag_got_end = false, flag_got_end_maybe = false;
    int i_end = ConstraintPoints::two_thirds_id(const_cast<Eigen::MatrixXd &>(init_points), touch_goal_);

    PtsChk_t pts_check;
    if (!computePointsToCheck(traj, i_end, pts_check))
      return CHK_RET::ERR;

    for (int i = 0; i < i_end; ++i)
    {
      for (size_t j = 0; j < pts_check[i].size(); ++j)
      {
        occ = grid_map_->getInflateOccupancy(pts_check[i][j].second);

        if (occ && !last_occ)
        {
          if (same_occ_state_times > ENOUGH_INTERVAL || i == 0)
          {
            in_id = i;
            flag_got_start = true;
          }
          same_occ_state_times = 0;
          flag_got_end_maybe = false;
        }
        else if (!occ && last_occ)
        {
          out_id = i + 1;
          flag_got_end_maybe = true;
          same_occ_state_times = 0;
        }
        else
        {
          ++same_occ_state_times;
        }

        if (flag_got_end_maybe && (same_occ_state_times > ENOUGH_INTERVAL || (i == i_end - 1)))
        {
          flag_got_end_maybe = false;
          flag_got_end = true;
        }

        last_occ = occ;

        if (flag_got_start && flag_got_end)
        {
          flag_got_start = false;
          flag_got_end = false;
          if (in_id < 0 || out_id < 0)
          {
            ROS_ERROR("Should not happen! in_id=%d, out_id=%d", in_id, out_id);
            return CHK_RET::ERR;
          }
          segment_ids.push_back(std::pair<int, int>(in_id, out_id));
        }
      }
    }

    if (segment_ids.size() == 0)
      return CHK_RET::OBS_FREE;

    /*** a star search ***/
    vector<vector<Eigen::Vector3d>> a_star_pathes;
    for (size_t i = 0; i < segment_ids.size(); ++i)
    {
      Eigen::Vector3d in(init_points.col(segment_ids[i].second)), out(init_points.col(segment_ids[i].first));
      ASTAR_RET ret = a_star_->AstarSearch(grid_map_->getResolution(), in, out);
      if (ret == ASTAR_RET::SUCCESS)
      {
        a_star_pathes.push_back(a_star_->getPath());
      }
      else if (ret == ASTAR_RET::SEARCH_ERR && i + 1 < segment_ids.size())
      {
        segment_ids[i].second = segment_ids[i + 1].second;
        segment_ids.erase(segment_ids.begin() + i + 1);
        --i;
        ROS_WARN("A corner case 2, I have never exeam it.");
      }
      else
      {
        ROS_WARN_COND(VERBOSE_OUTPUT, "A-star error, force return!");
        return CHK_RET::ERR;
      }
    }

    /*** calculate bounds ***/
    int id_low_bound, id_up_bound;
    vector<std::pair<int, int>> bounds(segment_ids.size());
    for (size_t i = 0; i < segment_ids.size(); i++)
    {
      if (i == 0)
      {
        id_low_bound = 1;
        if (segment_ids.size() > 1)
          id_up_bound = (int)(((segment_ids[0].second + segment_ids[1].first) - 1.0f) / 2);
        else
          id_up_bound = init_points.cols() - 2;
      }
      else if (i == segment_ids.size() - 1)
      {
        id_low_bound = (int)(((segment_ids[i].first + segment_ids[i - 1].second) + 1.0f) / 2);
        id_up_bound = init_points.cols() - 2;
      }
      else
      {
        id_low_bound = (int)(((segment_ids[i].first + segment_ids[i - 1].second) + 1.0f) / 2);
        id_up_bound = (int)(((segment_ids[i].second + segment_ids[i + 1].first) - 1.0f) / 2);
      }
      bounds[i] = std::pair<int, int>(id_low_bound, id_up_bound);
    }

    /*** Adjust segment length ***/
    vector<std::pair<int, int>> adjusted_segment_ids(segment_ids.size());
    constexpr double MINIMUM_PERCENT = 0.0;
    int minimum_points = round(init_points.cols() * MINIMUM_PERCENT), num_points;
    for (size_t i = 0; i < segment_ids.size(); i++)
    {
      num_points = segment_ids[i].second - segment_ids[i].first + 1;
      if (num_points < minimum_points)
      {
        double add_points_each_side = (int)(((minimum_points - num_points) + 1.0f) / 2);
        adjusted_segment_ids[i].first = segment_ids[i].first - add_points_each_side >= bounds[i].first
                                            ? segment_ids[i].first - add_points_each_side
                                            : bounds[i].first;
        adjusted_segment_ids[i].second = segment_ids[i].second + add_points_each_side <= bounds[i].second
                                             ? segment_ids[i].second + add_points_each_side
                                             : bounds[i].second;
      }
      else
      {
        adjusted_segment_ids[i].first = segment_ids[i].first;
        adjusted_segment_ids[i].second = segment_ids[i].second;
      }
    }

    for (size_t i = 1; i < adjusted_segment_ids.size(); i++)
    {
      if (adjusted_segment_ids[i - 1].second >= adjusted_segment_ids[i].first)
      {
        double middle = (double)(adjusted_segment_ids[i - 1].second + adjusted_segment_ids[i].first) / 2.0;
        adjusted_segment_ids[i - 1].second = static_cast<int>(middle - 0.1);
        adjusted_segment_ids[i].first = static_cast<int>(middle + 1.1);
      }
    }

    vector<std::pair<int, int>> final_segment_ids;

    /*** Assign data to each segment ***/
    for (size_t i = 0; i < segment_ids.size(); i++)
    {
      for (int j = adjusted_segment_ids[i].first; j <= adjusted_segment_ids[i].second; ++j)
        cps_.flag_temp[j] = false;

      int got_intersection_id = -1;
      for (int j = segment_ids[i].first + 1; j < segment_ids[i].second; ++j)
      {
        Eigen::Vector3d ctrl_pts_law(init_points.col(j + 1) - init_points.col(j - 1)), intersection_point;
        int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id;
        double val = (a_star_pathes[i][Astar_id] - init_points.col(j)).dot(ctrl_pts_law), init_val = val;
        while (true)
        {
          last_Astar_id = Astar_id;
          if (val >= 0)
          {
            ++Astar_id;
            if (Astar_id >= (int)a_star_pathes[i].size())
              break;
          }
          else
          {
            --Astar_id;
            if (Astar_id < 0)
              break;
          }
          val = (a_star_pathes[i][Astar_id] - init_points.col(j)).dot(ctrl_pts_law);
          if (val * init_val <= 0 && (abs(val) > 0 || abs(init_val) > 0))
          {
            intersection_point =
                a_star_pathes[i][Astar_id] +
                ((a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id]) *
                 (ctrl_pts_law.dot(init_points.col(j) - a_star_pathes[i][Astar_id]) / ctrl_pts_law.dot(a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id])));
            got_intersection_id = j;
            break;
          }
        }

        if (got_intersection_id >= 0)
        {
          double length = (intersection_point - init_points.col(j)).norm();
          if (length > 1e-5)
          {
            cps_.flag_temp[j] = true;
            for (double a = length; a >= 0.0; a -= grid_map_->getResolution())
            {
              bool occ_test = grid_map_->getInflateOccupancy((a / length) * intersection_point + (1 - a / length) * init_points.col(j));
              if (occ_test || a < grid_map_->getResolution())
              {
                if (occ_test)
                  a += grid_map_->getResolution();
                cps_.base_point[j].push_back((a / length) * intersection_point + (1 - a / length) * init_points.col(j));
                cps_.direction[j].push_back((intersection_point - init_points.col(j)).normalized());
                break;
              }
            }
          }
          else
            got_intersection_id = -1;
        }
      }

      /* Corner case */
      if (segment_ids[i].second - segment_ids[i].first == 1)
      {
        Eigen::Vector3d ctrl_pts_law(init_points.col(segment_ids[i].second) - init_points.col(segment_ids[i].first)), intersection_point;
        Eigen::Vector3d middle_point = (init_points.col(segment_ids[i].second) + init_points.col(segment_ids[i].first)) / 2;
        int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id;
        double val = (a_star_pathes[i][Astar_id] - middle_point).dot(ctrl_pts_law), init_val = val;
        while (true)
        {
          last_Astar_id = Astar_id;
          if (val >= 0)
          {
            ++Astar_id;
            if (Astar_id >= (int)a_star_pathes[i].size())
              break;
          }
          else
          {
            --Astar_id;
            if (Astar_id < 0)
              break;
          }
          val = (a_star_pathes[i][Astar_id] - middle_point).dot(ctrl_pts_law);
          if (val * init_val <= 0 && (abs(val) > 0 || abs(init_val) > 0))
          {
            intersection_point =
                a_star_pathes[i][Astar_id] +
                ((a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id]) *
                 (ctrl_pts_law.dot(middle_point - a_star_pathes[i][Astar_id]) / ctrl_pts_law.dot(a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id])));
            if ((intersection_point - middle_point).norm() > 0.01)
            {
              cps_.flag_temp[segment_ids[i].first] = true;
              cps_.base_point[segment_ids[i].first].push_back(init_points.col(segment_ids[i].first));
              cps_.direction[segment_ids[i].first].push_back((intersection_point - middle_point).normalized());
              got_intersection_id = segment_ids[i].first;
            }
            break;
          }
        }
      }

      if (got_intersection_id >= 0)
      {
        for (int j = got_intersection_id + 1; j <= adjusted_segment_ids[i].second; ++j)
          if (!cps_.flag_temp[j])
          {
            cps_.base_point[j].push_back(cps_.base_point[j - 1].back());
            cps_.direction[j].push_back(cps_.direction[j - 1].back());
          }
        for (int j = got_intersection_id - 1; j >= adjusted_segment_ids[i].first; --j)
          if (!cps_.flag_temp[j])
          {
            cps_.base_point[j].push_back(cps_.base_point[j + 1].back());
            cps_.direction[j].push_back(cps_.direction[j + 1].back());
          }
        final_segment_ids.push_back(adjusted_segment_ids[i]);
      }
    }

    segments = final_segment_ids;
    return CHK_RET::FINISH;
  }

  bool PolyTrajOptimizer::roughlyCheckConstraintPoints(void)
  {
    int in_id, out_id;
    vector<std::pair<int, int>> segment_ids;
    bool flag_new_obs_valid = false;
    int i_end = ConstraintPoints::two_thirds_id(cps_.points, touch_goal_);
    for (int i = 1; i <= i_end; ++i)
    {
      bool occ = grid_map_->getInflateOccupancy(cps_.points.col(i));

      if (occ)
      {
        for (size_t k = 0; k < cps_.direction[i].size(); ++k)
        {
          if ((cps_.points.col(i) - cps_.base_point[i][k]).dot(cps_.direction[i][k]) < 1 * grid_map_->getResolution())
          {
            occ = false;
            break;
          }
        }
      }

      if (occ)
      {
        flag_new_obs_valid = true;

        int j_inner;
        for (j_inner = i - 1; j_inner >= 0; --j_inner)
        {
          occ = grid_map_->getInflateOccupancy(cps_.points.col(j_inner));
          if (!occ)
          {
            in_id = j_inner;
            break;
          }
        }
        if (j_inner < 0)
        {
          ROS_ERROR("The drone is in obstacle. It means a crash in real-world.");
          in_id = 0;
        }

        for (j_inner = i + 1; j_inner < cps_.cp_size; ++j_inner)
        {
          occ = grid_map_->getInflateOccupancy(cps_.points.col(j_inner));
          if (!occ)
          {
            out_id = j_inner;
            break;
          }
        }
        if (j_inner >= cps_.cp_size)
        {
          ROS_WARN("Local target in collision, skip this planning.");
          force_stop_type_ = STOP_FOR_ERROR;
          return false;
        }

        i = j_inner + 1;
        segment_ids.push_back(std::pair<int, int>(in_id, out_id));
      }
    }

    if (flag_new_obs_valid)
    {
      vector<vector<Eigen::Vector3d>> a_star_pathes;
      for (size_t i = 0; i < segment_ids.size(); ++i)
      {
        Eigen::Vector3d in(cps_.points.col(segment_ids[i].second)), out(cps_.points.col(segment_ids[i].first));
        ASTAR_RET ret = a_star_->AstarSearch(grid_map_->getResolution(), in, out);
        if (ret == ASTAR_RET::SUCCESS)
          a_star_pathes.push_back(a_star_->getPath());
        else if (ret == ASTAR_RET::SEARCH_ERR && i + 1 < segment_ids.size())
        {
          segment_ids[i].second = segment_ids[i + 1].second;
          segment_ids.erase(segment_ids.begin() + i + 1);
          --i;
          ROS_WARN("A corner case 2, I have never exeam it.");
        }
        else
        {
          ROS_ERROR_COND(VERBOSE_OUTPUT, "A-star error");
          segment_ids.erase(segment_ids.begin() + i);
          --i;
        }
      }

      for (size_t i = 1; i < segment_ids.size(); i++)
      {
        if (segment_ids[i - 1].second >= segment_ids[i].first)
        {
          double middle = (double)(segment_ids[i - 1].second + segment_ids[i].first) / 2.0;
          segment_ids[i - 1].second = static_cast<int>(middle - 0.1);
          segment_ids[i].first = static_cast<int>(middle + 1.1);
        }
      }

      for (size_t i = 0; i < segment_ids.size(); ++i)
      {
        for (int j = segment_ids[i].first; j <= segment_ids[i].second; ++j)
          cps_.flag_temp[j] = false;

        int got_intersection_id = -1;
        for (int j = segment_ids[i].first + 1; j < segment_ids[i].second; ++j)
        {
          Eigen::Vector3d ctrl_pts_law(cps_.points.col(j + 1) - cps_.points.col(j - 1)), intersection_point;
          int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id;
          double val = (a_star_pathes[i][Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law), init_val = val;
          while (true)
          {
            last_Astar_id = Astar_id;
            if (val >= 0)
            {
              ++Astar_id;
              if (Astar_id >= (int)a_star_pathes[i].size())
                break;
            }
            else
            {
              --Astar_id;
              if (Astar_id < 0)
                break;
            }
            val = (a_star_pathes[i][Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law);
            if (val * init_val <= 0 && (abs(val) > 0 || abs(init_val) > 0))
            {
              intersection_point =
                  a_star_pathes[i][Astar_id] +
                  ((a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id]) *
                   (ctrl_pts_law.dot(cps_.points.col(j) - a_star_pathes[i][Astar_id]) / ctrl_pts_law.dot(a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id])));
              got_intersection_id = j;
              break;
            }
          }

          if (got_intersection_id >= 0)
          {
            double length = (intersection_point - cps_.points.col(j)).norm();
            if (length > 1e-5)
            {
              cps_.flag_temp[j] = true;
              for (double a = length; a >= 0.0; a -= grid_map_->getResolution())
              {
                bool occ_test = grid_map_->getInflateOccupancy((a / length) * intersection_point + (1 - a / length) * cps_.points.col(j));
                if (occ_test || a < grid_map_->getResolution())
                {
                  if (occ_test)
                    a += grid_map_->getResolution();
                  cps_.base_point[j].push_back((a / length) * intersection_point + (1 - a / length) * cps_.points.col(j));
                  cps_.direction[j].push_back((intersection_point - cps_.points.col(j)).normalized());
                  break;
                }
              }
            }
            else
              got_intersection_id = -1;
          }
        }

        if (got_intersection_id >= 0)
        {
          for (int j = got_intersection_id + 1; j <= segment_ids[i].second; ++j)
            if (!cps_.flag_temp[j])
            {
              cps_.base_point[j].push_back(cps_.base_point[j - 1].back());
              cps_.direction[j].push_back(cps_.direction[j - 1].back());
            }
          for (int j = got_intersection_id - 1; j >= segment_ids[i].first; --j)
            if (!cps_.flag_temp[j])
            {
              cps_.base_point[j].push_back(cps_.base_point[j + 1].back());
              cps_.direction[j].push_back(cps_.direction[j + 1].back());
            }
        }
        else
          ROS_WARN_COND(VERBOSE_OUTPUT, "Failed to generate direction. It doesn't matter.");
      }

      force_stop_type_ = STOP_FOR_REBOUND;
      return true;
    }

    return false;
  }

  bool PolyTrajOptimizer::allowRebound(void)
  {
    if (iter_num_ < 3)
      return false;

    double min_product = 1;
    for (int i = 3; i <= cps_.points.cols() - 4; ++i)
    {
      double product = ((cps_.points.col(i) - cps_.points.col(i - 1)).normalized()).dot((cps_.points.col(i + 1) - cps_.points.col(i)).normalized());
      if (product < min_product)
        min_product = product;
    }
    if (min_product < 0.87)
      return false;

    if (multitopology_data_.use_multitopology_trajs)
    {
      if (!multitopology_data_.initial_obstacles_avoided)
      {
        bool avoided = true;
        for (int i = 1; i < cps_.points.cols() - 1; ++i)
        {
          if (cps_.base_point[i].size() > 0)
          {
            if ((cps_.points.col(i) - cps_.base_point[i][0]).dot(cps_.direction[i][0]) < 0)
            {
              avoided = false;
              break;
            }
          }
        }
        multitopology_data_.initial_obstacles_avoided = avoided;
      }
      if (!multitopology_data_.initial_obstacles_avoided)
        return false;
    }

    return true;
  }

  std::vector<ConstraintPoints> PolyTrajOptimizer::distinctiveTrajs(vector<std::pair<int, int>> segments)
  {
    if (segments.size() == 0)
    {
      std::vector<ConstraintPoints> oneSeg;
      oneSeg.push_back(cps_);
      return oneSeg;
    }

    constexpr int MAX_TRAJS = 8;
    constexpr int VARIS = 2;
    int seg_upbound = std::min((int)segments.size(), static_cast<int>(floor(log(MAX_TRAJS) / log(VARIS))));
    std::vector<ConstraintPoints> control_pts_buf;
    control_pts_buf.reserve(MAX_TRAJS);
    const double RESOLUTION = grid_map_->getResolution();
    const double CTRL_PT_DIST = (cps_.points.col(0) - cps_.points.col(cps_.cp_size - 1)).norm() / (cps_.cp_size - 1);

    std::vector<std::pair<ConstraintPoints, ConstraintPoints>> RichInfoSegs;
    for (int i = 0; i < seg_upbound; i++)
    {
      std::pair<ConstraintPoints, ConstraintPoints> RichInfoOneSeg;
      ConstraintPoints RichInfoOneSeg_temp;
      cps_.segment(RichInfoOneSeg_temp, segments[i].first, segments[i].second);
      RichInfoOneSeg.first = RichInfoOneSeg_temp;
      RichInfoOneSeg.second = RichInfoOneSeg_temp;
      RichInfoSegs.push_back(RichInfoOneSeg);
    }

    for (int i = 0; i < seg_upbound; i++)
    {
      if (RichInfoSegs[i].first.cp_size > 1)
      {
        int occ_start_id = -1, occ_end_id = -1;
        Eigen::Vector3d occ_start_pt, occ_end_pt;
        for (int j = 0; j < RichInfoSegs[i].first.cp_size - 1; j++)
        {
          double step_size = RESOLUTION / (RichInfoSegs[i].first.points.col(j) - RichInfoSegs[i].first.points.col(j + 1)).norm() / 2;
          for (double a = 1; a > 0; a -= step_size)
          {
            Eigen::Vector3d pt(a * RichInfoSegs[i].first.points.col(j) + (1 - a) * RichInfoSegs[i].first.points.col(j + 1));
            if (grid_map_->getInflateOccupancy(pt))
            {
              occ_start_id = j;
              occ_start_pt = pt;
              goto exit_multi_loop1;
            }
          }
        }
      exit_multi_loop1:;
        for (int j = RichInfoSegs[i].first.cp_size - 1; j >= 1; j--)
        {
          double step_size = RESOLUTION / (RichInfoSegs[i].first.points.col(j) - RichInfoSegs[i].first.points.col(j - 1)).norm();
          for (double a = 1; a > 0; a -= step_size)
          {
            Eigen::Vector3d pt(a * RichInfoSegs[i].first.points.col(j) + (1 - a) * RichInfoSegs[i].first.points.col(j - 1));
            if (grid_map_->getInflateOccupancy(pt))
            {
              occ_end_id = j;
              occ_end_pt = pt;
              goto exit_multi_loop2;
            }
          }
        }
      exit_multi_loop2:;

        if (occ_start_id == -1 || occ_end_id == -1)
        {
          segments.erase(segments.begin() + i);
          RichInfoSegs.erase(RichInfoSegs.begin() + i);
          seg_upbound--;
          i--;
          continue;
        }

        for (int j = occ_start_id; j <= occ_end_id; j++)
        {
          Eigen::Vector3d base_pt_reverse, base_vec_reverse;
          if (RichInfoSegs[i].first.base_point[j].size() != 1)
          {
            ROS_ERROR("Wrong number of base_points!!! Should not be happen!.");
            std::vector<ConstraintPoints> blank;
            return blank;
          }

          base_vec_reverse = -RichInfoSegs[i].first.direction[j][0];

          if (j == occ_start_id)
            base_pt_reverse = occ_start_pt;
          else if (j == occ_end_id)
            base_pt_reverse = occ_end_pt;
          else
            base_pt_reverse = RichInfoSegs[i].first.points.col(j) + base_vec_reverse * (RichInfoSegs[i].first.base_point[j][0] - RichInfoSegs[i].first.points.col(j)).norm();

          if (grid_map_->getInflateOccupancy(base_pt_reverse))
          {
            double l_upbound = 5 * CTRL_PT_DIST;
            double l = RESOLUTION;
            for (; l <= l_upbound; l += RESOLUTION)
            {
              Eigen::Vector3d base_pt_temp = base_pt_reverse + l * base_vec_reverse;
              if (!grid_map_->getInflateOccupancy(base_pt_temp))
              {
                RichInfoSegs[i].second.base_point[j][0] = base_pt_temp;
                RichInfoSegs[i].second.direction[j][0] = base_vec_reverse;
                break;
              }
            }
            if (l > l_upbound)
            {
              ROS_WARN_COND(VERBOSE_OUTPUT, "Can't find the new base points at the opposite within the threshold. i=%d, j=%d", i, j);
              segments.erase(segments.begin() + i);
              RichInfoSegs.erase(RichInfoSegs.begin() + i);
              seg_upbound--;
              i--;
              goto exit_multi_loop3;
            }
          }
          else if ((base_pt_reverse - RichInfoSegs[i].first.points.col(j)).norm() >= RESOLUTION)
          {
            RichInfoSegs[i].second.base_point[j][0] = base_pt_reverse;
            RichInfoSegs[i].second.direction[j][0] = base_vec_reverse;
          }
          else
          {
            ROS_WARN_COND(VERBOSE_OUTPUT, "base_point and control point are too close!");
            segments.erase(segments.begin() + i);
            RichInfoSegs.erase(RichInfoSegs.begin() + i);
            seg_upbound--;
            i--;
            goto exit_multi_loop3;
          }
        }

        if (RichInfoSegs[i].second.cp_size)
        {
          for (int j = occ_start_id - 1; j >= 0; j--)
          {
            RichInfoSegs[i].second.base_point[j][0] = RichInfoSegs[i].second.base_point[occ_start_id][0];
            RichInfoSegs[i].second.direction[j][0] = RichInfoSegs[i].second.direction[occ_start_id][0];
          }
          for (int j = occ_end_id + 1; j < RichInfoSegs[i].second.cp_size; j++)
          {
            RichInfoSegs[i].second.base_point[j][0] = RichInfoSegs[i].second.base_point[occ_end_id][0];
            RichInfoSegs[i].second.direction[j][0] = RichInfoSegs[i].second.direction[occ_end_id][0];
          }
        }

      exit_multi_loop3:;
      }
      else
      {
        Eigen::Vector3d base_vec_reverse = -RichInfoSegs[i].first.direction[0][0];
        Eigen::Vector3d base_pt_reverse = RichInfoSegs[i].first.points.col(0) + base_vec_reverse * (RichInfoSegs[i].first.base_point[0][0] - RichInfoSegs[i].first.points.col(0)).norm();

        if (grid_map_->getInflateOccupancy(base_pt_reverse))
        {
          double l_upbound = 5 * CTRL_PT_DIST;
          double l = RESOLUTION;
          for (; l <= l_upbound; l += RESOLUTION)
          {
            Eigen::Vector3d base_pt_temp = base_pt_reverse + l * base_vec_reverse;
            if (!grid_map_->getInflateOccupancy(base_pt_temp))
            {
              RichInfoSegs[i].second.base_point[0][0] = base_pt_temp;
              RichInfoSegs[i].second.direction[0][0] = base_vec_reverse;
              break;
            }
          }
          if (l > l_upbound)
          {
            segments.erase(segments.begin() + i);
            RichInfoSegs.erase(RichInfoSegs.begin() + i);
            seg_upbound--;
            i--;
          }
        }
        else if ((base_pt_reverse - RichInfoSegs[i].first.points.col(0)).norm() >= RESOLUTION)
        {
          RichInfoSegs[i].second.base_point[0][0] = base_pt_reverse;
          RichInfoSegs[i].second.direction[0][0] = base_vec_reverse;
        }
        else
        {
          segments.erase(segments.begin() + i);
          RichInfoSegs.erase(RichInfoSegs.begin() + i);
          seg_upbound--;
          i--;
        }
      }
    }

    if (seg_upbound == 0)
    {
      std::vector<ConstraintPoints> oneSeg;
      oneSeg.push_back(cps_);
      return oneSeg;
    }

    std::vector<int> selection(seg_upbound);
    std::fill(selection.begin(), selection.end(), 0);
    selection[0] = -1;
    int max_traj_nums = static_cast<int>(pow(VARIS, seg_upbound));
    for (int i = 0; i < max_traj_nums; i++)
    {
      int digit_id = 0;
      selection[digit_id]++;
      while (digit_id < seg_upbound && selection[digit_id] >= VARIS)
      {
        selection[digit_id] = 0;
        digit_id++;
        if (digit_id >= seg_upbound)
        {
          ROS_ERROR("Should not happen!!! digit_id=%d, seg_upbound=%d", digit_id, seg_upbound);
        }
        selection[digit_id]++;
      }

      ConstraintPoints cpsOneSample;
      cpsOneSample.resize_cp(cps_.cp_size);
      int cp_id = 0, seg_id = 0, cp_of_seg_id = 0;
      while (cp_id < cps_.cp_size)
      {
        if (seg_id >= seg_upbound || cp_id < segments[seg_id].first || cp_id > segments[seg_id].second)
        {
          cpsOneSample.points.col(cp_id) = cps_.points.col(cp_id);
          cpsOneSample.base_point[cp_id] = cps_.base_point[cp_id];
          cpsOneSample.direction[cp_id] = cps_.direction[cp_id];
        }
        else if (cp_id >= segments[seg_id].first && cp_id <= segments[seg_id].second)
        {
          if (!selection[seg_id])
          {
            cpsOneSample.points.col(cp_id) = RichInfoSegs[seg_id].first.points.col(cp_of_seg_id);
            cpsOneSample.base_point[cp_id] = RichInfoSegs[seg_id].first.base_point[cp_of_seg_id];
            cpsOneSample.direction[cp_id] = RichInfoSegs[seg_id].first.direction[cp_of_seg_id];
            cp_of_seg_id++;
          }
          else
          {
            if (RichInfoSegs[seg_id].second.cp_size)
            {
              cpsOneSample.points.col(cp_id) = RichInfoSegs[seg_id].second.points.col(cp_of_seg_id);
              cpsOneSample.base_point[cp_id] = RichInfoSegs[seg_id].second.base_point[cp_of_seg_id];
              cpsOneSample.direction[cp_id] = RichInfoSegs[seg_id].second.direction[cp_of_seg_id];
              cp_of_seg_id++;
            }
            else
              goto abandon_this_trajectory;
          }

          if (cp_id == segments[seg_id].second)
          {
            cp_of_seg_id = 0;
            seg_id++;
          }
        }
        else
        {
          ROS_ERROR("Should not happen!!!!");
        }

        cp_id++;
      }

      control_pts_buf.push_back(cpsOneSample);

    abandon_this_trajectory:;
    }

    return control_pts_buf;
  }

  void PolyTrajOptimizer::distanceSqrVarianceWithGradCost2p(const Eigen::MatrixXd &ps,
                                                            Eigen::MatrixXd &gdp,
                                                            double &var)
  {
    int N = ps.cols() - 1;
    Eigen::MatrixXd dps = ps.rightCols(N) - ps.leftCols(N);
    Eigen::VectorXd dsqrs = dps.colwise().squaredNorm().transpose();
    double dquarsum = dsqrs.squaredNorm();
    double dquarmean = dquarsum / N;
    var = wei_sqrvar_ * (dquarmean);
    gdp.resize(3, N + 1);
    gdp.setZero();
    for (int i = 0; i <= N; i++)
    {
      if (i != 0)
        gdp.col(i) += wei_sqrvar_ * (4.0 * (dsqrs(i - 1)) / N * dps.col(i - 1));
      if (i != N)
        gdp.col(i) += wei_sqrvar_ * (-4.0 * (dsqrs(i)) / N * dps.col(i));
    }
  }

  /* helper functions */
  void PolyTrajOptimizer::setParam(ros::NodeHandle &nh)
  {
    nh.param("optimization/constraint_points_perPiece", cps_num_prePiece_, -1);
    nh.param("optimization/weight_obstacle", wei_obs_, -1.0);
    nh.param("optimization/weight_obstacle_soft", wei_obs_soft_, -1.0);
    nh.param("optimization/weight_swarm", wei_swarm_, -1.0);
    nh.param("optimization/weight_feasibility", wei_feas_, -1.0);
    nh.param("optimization/weight_sqrvariance", wei_sqrvar_, -1.0);
    nh.param("optimization/weight_time", wei_time_, -1.0);
    nh.param("optimization/obstacle_clearance", obs_clearance_, -1.0);
    nh.param("optimization/obstacle_clearance_soft", obs_clearance_soft_, -1.0);
    nh.param("optimization/swarm_clearance", swarm_clearance_, -1.0);
    nh.param("optimization/max_vel", max_vel_, -1.0);
    nh.param("optimization/max_acc", max_acc_, -1.0);
    nh.param("optimization/max_jer", max_jer_, -1.0);
  }

  void PolyTrajOptimizer::setEnvironment(const GridMap::Ptr &map)
  {
    grid_map_ = map;
    a_star_.reset(new AStar);
    a_star_->initGridMap(grid_map_, Eigen::Vector3i(100, 100, 100));
  }

  void PolyTrajOptimizer::setControlPoints(const Eigen::MatrixXd &points)
  {
    cps_.points = points;
  }

  void PolyTrajOptimizer::setSwarmTrajs(SwarmTrajData *swarm_trajs_ptr) { swarm_trajs_ = swarm_trajs_ptr; }
  void PolyTrajOptimizer::setDroneId(const int drone_id) { drone_id_ = drone_id; }
  void PolyTrajOptimizer::setIfTouchGoal(const bool touch_goal) { touch_goal_ = touch_goal; }
  void PolyTrajOptimizer::setConstraintPoints(ConstraintPoints cps) { cps_ = cps; }
  void PolyTrajOptimizer::setUseMultitopologyTrajs(bool use_multitopology_trajs) { multitopology_data_.use_multitopology_trajs = use_multitopology_trajs; }

} // namespace ego_planner
