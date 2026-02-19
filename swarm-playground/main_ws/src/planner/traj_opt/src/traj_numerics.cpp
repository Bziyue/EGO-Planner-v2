#include "optimizer/poly_traj_optimizer.h"

using namespace std;

namespace ego_planner
{
  // =====================================================
  //  Helpers for discrete variance gradient (same-iteration)
  // =====================================================
  struct ZeroTimeCostFunction
  {
    double operator()(const std::vector<double> &, Eigen::VectorXd &) const
    {
      return 0.0;
    }
  };

  class VarianceGradCostFunction
  {
  public:
    const Eigen::MatrixXd *variance_grad{nullptr};
    const std::vector<double> *segment_dt{nullptr};
    int cps_per_piece{1};

    mutable int current_seg_{-1};
    mutable int step_in_seg_{0};

    double operator()(double /*t*/, double /*t_global*/, int seg_idx,
                      const Vec3 & /*p*/, const Vec3 & /*v*/,
                      const Vec3 & /*a*/, const Vec3 & /*j*/, const Vec3 & /*s*/,
                      Vec3 &gp, Vec3 & /*gv*/, Vec3 & /*ga*/,
                      Vec3 & /*gj*/, Vec3 & /*gs*/, double & /*gt*/) const
    {
      if (seg_idx != current_seg_)
      {
        current_seg_ = seg_idx;
        step_in_seg_ = 0;
      }

      int cp_idx = seg_idx * cps_per_piece + step_in_seg_;
      if (variance_grad && segment_dt &&
          cp_idx < variance_grad->cols() &&
          seg_idx >= 0 && seg_idx < (int)segment_dt->size())
      {
        double dt = (*segment_dt)[seg_idx];
        if (dt > 1e-12)
        {
          gp += variance_grad->col(cp_idx) / dt;
        }
      }

      ++step_in_seg_;
      return 0.0;
    }
  };

  // =====================================================
  //  Generate trajectory from states using QuinticSplineND
  // =====================================================
  PPoly3D PolyTrajOptimizer::generateTrajectory(
      const Eigen::MatrixXd &iniState, const Eigen::MatrixXd &finState,
      const Eigen::MatrixXd &innerPts, const Eigen::VectorXd &durations)
  {
    int piece_num = durations.size();
    // Build waypoints: start + inner + end
    WaypointsMat waypoints(innerPts.cols() + 2, 3);
    waypoints.row(0) = iniState.col(0).transpose();
    for (int i = 0; i < innerPts.cols(); ++i)
      waypoints.row(i + 1) = innerPts.col(i).transpose();
    waypoints.row(innerPts.cols() + 1) = finState.col(0).transpose();

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

    for (int i = 0; i < N; ++i)
    {
      double dur = durations(i);
      double step = dur / K;
      for (int j = 0; j <= K; ++j)
      {
        double t = t_accum + step * j;
        cstr_pts.col(idx) = traj.evaluate(t, SplineTrajectory::Deriv::Pos);
        if (j != K || (j == K && i == N - 1))
          ++idx;
      }
      t_accum += dur;
    }

    return cstr_pts;
  }

  // =====================================================
  //  LBFGS cost function callback
  // =====================================================
  double PolyTrajOptimizer::costFunctionCallback(void *func_data, const double *x, double *grad, const int n)
  {
    PolyTrajOptimizer *opt = reinterpret_cast<PolyTrajOptimizer *>(func_data);

    fill(opt->min_ellip_dist2_.begin(), opt->min_ellip_dist2_.end(), std::numeric_limits<double>::max());

    Eigen::VectorXd x_vec = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::VectorXd grad_vec = Eigen::VectorXd::Zero(n);

    // Pre-compute segment dt for variance gradient scaling.
    // x layout: [tau_0, ..., tau_{N-1}, P_inner, ...]
    // T_i = QuadInvTimeMap::toTime(tau_i), dt_i = T_i / K
    {
      SplineTrajectory::QuadInvTimeMap time_map;
      opt->integral_cost_func_.segment_dt_.resize(opt->piece_num_);
      for (int i = 0; i < opt->piece_num_; ++i)
      {
        double T = time_map.toTime(x_vec(i));
        opt->integral_cost_func_.segment_dt_[i] = T / opt->cps_num_prePiece_;
      }
    }

    opt->integral_cost_func_.resetAccumulation();

    double total_cost = opt->splineOpt_.evaluate(
        x_vec, grad_vec,
        opt->time_cost_func_,
        opt->integral_cost_func_);

    // Distance variance cost on constraint points (post-processing).
    // Apply its gradient in the SAME iteration to keep L-BFGS consistent.
    if (opt->wei_sqrvar_ > 0 && opt->cps_.cp_size > 1)
    {
      Eigen::MatrixXd gdp;
      double var_cost_value = 0.0;
      opt->distanceSqrVarianceWithGradCost2p(opt->cps_.points, gdp, var_cost_value);
      total_cost += var_cost_value;

      // Inject discrete variance gradient via a second pass (no time/energy cost).
      VarianceGradCostFunction var_cost_func;
      var_cost_func.variance_grad = &gdp;
      var_cost_func.segment_dt = &opt->integral_cost_func_.segment_dt_;
      var_cost_func.cps_per_piece = opt->cps_num_prePiece_;

      Eigen::VectorXd grad_var = Eigen::VectorXd::Zero(n);
      ZeroTimeCostFunction zero_time;
      double rho_backup = opt->rho_energy_;
      opt->splineOpt_.setEnergyWeights(0.0);
      opt->splineOpt_.evaluate(x_vec, grad_var, zero_time, var_cost_func);
      opt->splineOpt_.setEnergyWeights(rho_backup);

      grad_vec += grad_var;
    }

    // Copy gradients back to raw pointer
    Eigen::Map<Eigen::VectorXd>(grad, n) = grad_vec;

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
  //  Dense point sampling for collision checking
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

  // =====================================================
  //  Distance variance cost on constraint points
  // =====================================================
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

} // namespace ego_planner
