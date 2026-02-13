#ifndef _POLY_TRAJ_OPTIMIZER_H_
#define _POLY_TRAJ_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <path_searching/dyn_a_star.h>
#include <plan_env/grid_map.h>
#include <ros/ros.h>
#include "optimizer/lbfgs.hpp"
#include <traj_utils/plan_container.hpp>
#include "SplineTrajectory/SplineTrajectory.hpp"
#include "SplineTrajectory/SplineOptimizer.hpp"

namespace ego_planner
{
  // =====================================================
  //  Type aliases for the Spline-based trajectory system
  // =====================================================
  constexpr int TRAJ_DIM = 3;
  using SplineOpt = SplineTrajectory::SplineOptimizer<TRAJ_DIM,
                                                       SplineTrajectory::QuinticSplineND<TRAJ_DIM>,
                                                       SplineTrajectory::QuadInvTimeMap>;
  using SplineTraj = SplineTrajectory::QuinticSplineND<TRAJ_DIM>;
  using PPoly3D = SplineTrajectory::PPolyND<TRAJ_DIM>;
  using Vec3 = Eigen::Vector3d;
  using BCs = SplineTrajectory::BoundaryConditions<TRAJ_DIM>;
  using WaypointsVec = SplineTrajectory::SplineVector<Eigen::Vector3d>;

  // =====================================================
  //  ConstraintPoints (unchanged from original)
  // =====================================================
  class ConstraintPoints
  {
  public:
    int cp_size; // deformation points
    Eigen::MatrixXd points;
    std::vector<std::vector<Eigen::Vector3d>> base_point;
    std::vector<std::vector<Eigen::Vector3d>> direction;
    std::vector<bool> flag_temp;

    void resize_cp(const int size_set)
    {
      cp_size = size_set;
      base_point.clear();
      direction.clear();
      flag_temp.clear();
      points.resize(3, size_set);
      base_point.resize(cp_size);
      direction.resize(cp_size);
      flag_temp.resize(cp_size);
    }

    void segment(ConstraintPoints &buf, const int start, const int end)
    {
      if (start < 0 || end >= cp_size || points.rows() != 3)
      {
        ROS_ERROR("Wrong segment index! start=%d, end=%d", start, end);
        return;
      }
      buf.resize_cp(end - start + 1);
      buf.points = points.block(0, start, 3, end - start + 1);
      buf.cp_size = end - start + 1;
      for (int i = start; i <= end; i++)
      {
        buf.base_point[i - start] = base_point[i];
        buf.direction[i - start] = direction[i];
      }
    }

    static inline int two_thirds_id(Eigen::MatrixXd &points, const bool touch_goal)
    {
      return touch_goal ? points.cols() - 1 : points.cols() - 1 - (points.cols() - 2) / 3;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  // =====================================================
  //  Cost function classes (modular, injectable into SplineOptimizer)
  // =====================================================

  /**
   * @brief TimeCost: penalizes total trajectory time
   * Conforms to SplineOptimizer::TimeCostProtocol
   */
  struct TimeCostFunction
  {
    double wei_time;

    TimeCostFunction() : wei_time(0.0) {}

    double operator()(const std::vector<double> &Ts, Eigen::VectorXd &grad) const
    {
      double cost = 0.0;
      for (int i = 0; i < (int)Ts.size(); ++i)
      {
        grad(i) += wei_time;
        cost += Ts[i] * wei_time;
      }
      return cost;
    }
  };

  /**
   * @brief Integral cost combining obstacle, swarm, feasibility, and distance variance costs.
   * Conforms to SplineOptimizer::IntegralCostProtocol
   */
  class IntegralCostFunction
  {
  public:
    // Grid map for obstacle checking
    GridMap::Ptr grid_map;
    // Constraint points reference
    ConstraintPoints *cps;
    // Swarm trajectory data
    SwarmTrajData *swarm_trajs;
    // Parameters
    double wei_obs, wei_obs_soft, wei_swarm, wei_feas, wei_sqrvar;
    double obs_clearance, obs_clearance_soft, swarm_clearance;
    double max_vel, max_acc, max_jer;
    int drone_id;
    double t_now;
    bool touch_goal;
    int cps_per_piece; // K: number of constraint points per segment
    // Mutable state for tracking
    mutable int current_seg_;   // current segment index
    mutable int step_in_seg_;   // step within current segment (0..K)
    mutable std::vector<double> *min_ellip_dist2_ptr;
    // Accumulated costs for reporting
    mutable Eigen::VectorXd accumulated_costs; // [obs, swarm, feas, sqrvar]

    IntegralCostFunction()
        : grid_map(nullptr), cps(nullptr), swarm_trajs(nullptr),
          wei_obs(0), wei_obs_soft(0), wei_swarm(0), wei_feas(0), wei_sqrvar(0),
          obs_clearance(0), obs_clearance_soft(0), swarm_clearance(0),
          max_vel(0), max_acc(0), max_jer(0),
          drone_id(-1), t_now(0), touch_goal(false), cps_per_piece(5),
          current_seg_(-1), step_in_seg_(0),
          min_ellip_dist2_ptr(nullptr)
    {
      accumulated_costs.resize(4);
      accumulated_costs.setZero();
    }

    void resetAccumulation() const
    {
      current_seg_ = -1;
      step_in_seg_ = 0;
      accumulated_costs.setZero();
    }

    double operator()(double t, double t_global, int seg_idx,
                      const Vec3 &p, const Vec3 &v,
                      const Vec3 &a, const Vec3 &j, const Vec3 &s,
                      Vec3 &gp, Vec3 &gv, Vec3 &ga,
                      Vec3 &gj, Vec3 &gs, double &gt) const
    {
      double cost = 0.0;

      // Compute correct constraint point index:
      // SplineOptimizer evaluates K+1 points per segment (k=0..K).
      // Constraint points are indexed as seg_idx * K + step_in_seg.
      // At boundaries, k=K of seg i and k=0 of seg i+1 map to the same cp index.
      if (seg_idx != current_seg_)
      {
        current_seg_ = seg_idx;
        step_in_seg_ = 0;
      }

      int cp_idx = seg_idx * cps_per_piece + step_in_seg_;

      // Update constraint point position
      if (cps && cp_idx < cps->cp_size)
        cps->points.col(cp_idx) = p;

      // --- Obstacle cost ---
      cost += obstacleGradCostP(cp_idx, p, gp);

      // --- Swarm cost ---
      cost += swarmGradCostP(cp_idx, t_global, p, v, gp, gt);

      // --- Feasibility costs (vel, acc, jerk) ---
      cost += feasibilityGradCost(v, a, j, gv, ga, gj);

      ++step_in_seg_;
      return cost;
    }

  private:
    double obstacleGradCostP(int cp_idx, const Vec3 &p, Vec3 &gradp) const
    {
      if (!cps || cp_idx == 0 || cp_idx >= cps->cp_size ||
          cp_idx > ConstraintPoints::two_thirds_id(cps->points, touch_goal))
        return 0.0;

      double costp = 0.0;
      for (size_t k = 0; k < cps->direction[cp_idx].size(); ++k)
      {
        Vec3 ray = (p - cps->base_point[cp_idx][k]);
        double dist = ray.dot(cps->direction[cp_idx][k]);
        double dist_err = obs_clearance - dist;
        double dist_err_soft = obs_clearance_soft - dist;
        Vec3 dist_grad = cps->direction[cp_idx][k];

        if (dist_err > 0)
        {
          costp += wei_obs * pow(dist_err, 3);
          gradp += -wei_obs * 3.0 * dist_err * dist_err * dist_grad;
        }

        if (dist_err_soft > 0)
        {
          double r = 0.05;
          double rsqr = r * r;
          double term = sqrt(1.0 + dist_err_soft * dist_err_soft / rsqr);
          costp += wei_obs_soft * rsqr * (term - 1.0);
          gradp += -wei_obs_soft * dist_err_soft / term * dist_grad;
        }
      }
      accumulated_costs(0) += costp;
      return costp;
    }

    double swarmGradCostP(int cp_idx, double t_global, const Vec3 &p, const Vec3 &v,
                          Vec3 &gradp, double &gt) const
    {
      if (!swarm_trajs || !cps || cp_idx <= 0 || cp_idx >= cps->cp_size ||
          cp_idx > ConstraintPoints::two_thirds_id(cps->points, touch_goal))
        return 0.0;

      double costp = 0.0;
      constexpr double a_param = 2.0, b_param = 1.0;
      constexpr double inv_a2 = 1.0 / (a_param * a_param), inv_b2 = 1.0 / (b_param * b_param);

      for (size_t id = 0; id < swarm_trajs->size(); id++)
      {
        if ((swarm_trajs->at(id).drone_id < 0) || swarm_trajs->at(id).drone_id == drone_id)
          continue;

        double traj_i_start_time = swarm_trajs->at(id).start_time;
        double pt_time = (t_now - traj_i_start_time) + t_global;
        const double CLEARANCE = (swarm_clearance + swarm_trajs->at(id).des_clearance) * 1.5;
        const double CLEARANCE2 = CLEARANCE * CLEARANCE;

        Vec3 swarm_p, swarm_v;
        if (pt_time < swarm_trajs->at(id).duration)
        {
          swarm_p = swarm_trajs->at(id).traj.evaluate(swarm_trajs->at(id).traj.getStartTime() + pt_time, SplineTrajectory::Deriv::Pos);
          swarm_v = swarm_trajs->at(id).traj.evaluate(swarm_trajs->at(id).traj.getStartTime() + pt_time, SplineTrajectory::Deriv::Vel);
        }
        else
        {
          double end_t = swarm_trajs->at(id).traj.getStartTime() + swarm_trajs->at(id).duration;
          swarm_v = swarm_trajs->at(id).traj.evaluate(end_t, SplineTrajectory::Deriv::Vel);
          swarm_p = swarm_trajs->at(id).traj.evaluate(end_t, SplineTrajectory::Deriv::Pos) +
                    (pt_time - swarm_trajs->at(id).duration) * swarm_v;
        }
        Vec3 dist_vec = p - swarm_p;
        double ellip_dist2 = dist_vec(2) * dist_vec(2) * inv_a2 + (dist_vec(0) * dist_vec(0) + dist_vec(1) * dist_vec(1)) * inv_b2;
        double dist2_err = CLEARANCE2 - ellip_dist2;
        double dist2_err2 = dist2_err * dist2_err;
        double dist2_err3 = dist2_err2 * dist2_err;

        if (dist2_err3 > 0)
        {
          costp += wei_swarm * dist2_err3;
          Vec3 dJ_dP = wei_swarm * 3 * dist2_err2 * (-2) * Vec3(inv_b2 * dist_vec(0), inv_b2 * dist_vec(1), inv_a2 * dist_vec(2));
          gradp += dJ_dP;
          gt += dJ_dP.dot(v - swarm_v);
        }

        if (min_ellip_dist2_ptr && id < min_ellip_dist2_ptr->size())
        {
          if ((*min_ellip_dist2_ptr)[id] > ellip_dist2)
            (*min_ellip_dist2_ptr)[id] = ellip_dist2;
        }
      }
      accumulated_costs(1) += costp;
      return costp;
    }

    double feasibilityGradCost(const Vec3 &v, const Vec3 &a, const Vec3 &j,
                               Vec3 &gv, Vec3 &ga, Vec3 &gj) const
    {
      double cost = 0.0;

      // Velocity
      double vpen = v.squaredNorm() - max_vel * max_vel;
      if (vpen > 0)
      {
        gv += wei_feas * 6 * vpen * vpen * v;
        cost += wei_feas * vpen * vpen * vpen;
      }

      // Acceleration
      double apen = a.squaredNorm() - max_acc * max_acc;
      if (apen > 0)
      {
        ga += wei_feas * 6 * apen * apen * a;
        cost += wei_feas * apen * apen * apen;
      }

      // Jerk
      double jpen = j.squaredNorm() - max_jer * max_jer;
      if (jpen > 0)
      {
        gj += wei_feas * 6 * jpen * jpen * j;
        cost += wei_feas * jpen * jpen * jpen;
      }

      accumulated_costs(2) += cost;
      return cost;
    }
  };

  // =====================================================
  //  PolyTrajOptimizer (using SplineOptimizer)
  // =====================================================
  class PolyTrajOptimizer
  {

  private:
    GridMap::Ptr grid_map_;
    AStar::Ptr a_star_;

    // SplineOptimizer replaces MinJerkOpt
    SplineOpt splineOpt_;

    SwarmTrajData *swarm_trajs_{NULL};
    ConstraintPoints cps_;

    int drone_id_;
    int cps_num_prePiece_;
    int variable_num_;
    int piece_num_;
    int iter_num_;
    std::vector<double> min_ellip_dist2_;
    bool touch_goal_;
    struct MultitopologyData_t
    {
      bool use_multitopology_trajs{false};
      bool initial_obstacles_avoided{false};
    } multitopology_data_;

    enum FORCE_STOP_OPTIMIZE_TYPE
    {
      DONT_STOP,
      STOP_FOR_REBOUND,
      STOP_FOR_ERROR
    } force_stop_type_;

    /* optimization parameters */
    double wei_obs_, wei_obs_soft_;
    double wei_swarm_, wei_swarm_mod_;
    double wei_feas_;
    double wei_sqrvar_;
    double wei_time_;
    double obs_clearance_, obs_clearance_soft_, swarm_clearance_;
    double max_vel_, max_acc_, max_jer_;

    double t_now_;

    // Cost function instances
    TimeCostFunction time_cost_func_;
    IntegralCostFunction integral_cost_func_;

  public:
    PolyTrajOptimizer() {}
    ~PolyTrajOptimizer() {}

    enum CHK_RET
    {
      OBS_FREE,
      ERR,
      FINISH
    };

    /* set variables */
    void setParam(ros::NodeHandle &nh);
    void setEnvironment(const GridMap::Ptr &map);
    void setControlPoints(const Eigen::MatrixXd &points);
    void setSwarmTrajs(SwarmTrajData *swarm_trajs_ptr);
    void setDroneId(const int drone_id);
    void setIfTouchGoal(const bool touch_goal);
    void setConstraintPoints(ConstraintPoints cps);
    void setUseMultitopologyTrajs(bool use_multitopology_trajs);

    /* helper functions */
    inline const ConstraintPoints &getControlPoints(void) { return cps_; }
    inline const SplineOpt &getSplineOpt(void) const { return splineOpt_; }
    inline int get_cps_num_prePiece_(void) { return cps_num_prePiece_; }
    inline double get_swarm_clearance_(void) { return swarm_clearance_; }

    /**
     * @brief Generate a trajectory using SplineOptimizer from init states.
     * Returns a PPoly3D trajectory that can be used downstream.
     */
    PPoly3D generateTrajectory(
        const Eigen::MatrixXd &iniState, const Eigen::MatrixXd &finState,
        const Eigen::MatrixXd &innerPts, const Eigen::VectorXd &durations);

    /**
     * @brief Get initial constraint points from the spline trajectory.
     */
    Eigen::MatrixXd getInitConstraintPoints(const PPoly3D &traj,
                                             const Eigen::VectorXd &durations,
                                             int K) const;

    /* main planning API */
    bool optimizeTrajectory(const Eigen::MatrixXd &iniState, const Eigen::MatrixXd &finState,
                            const Eigen::MatrixXd &initInnerPts, const Eigen::VectorXd &initT,
                            double &final_cost);

    bool computePointsToCheck(const PPoly3D &traj, int id_end, PtsChk_t &pts_check);

    std::vector<std::pair<int, int>> finelyCheckConstraintPointsOnly(Eigen::MatrixXd &init_points);

    CHK_RET finelyCheckAndSetConstraintPoints(std::vector<std::pair<int, int>> &segments,
                                              const PPoly3D &traj,
                                              const Eigen::MatrixXd &init_points,
                                              const bool flag_first_init /*= true*/);

    bool roughlyCheckConstraintPoints(void);

    bool allowRebound(void);

    /* multi-topo support */
    std::vector<ConstraintPoints> distinctiveTrajs(std::vector<std::pair<int, int>> segments);

    /* distance variance cost (applied as post-processing on constraint points) */
    void distanceSqrVarianceWithGradCost2p(const Eigen::MatrixXd &ps,
                                           Eigen::MatrixXd &gdp,
                                           double &var);

  private:
    /* callbacks by the L-BFGS optimizer */
    static double costFunctionCallback(void *func_data, const double *x, double *grad, const int n);

    static int earlyExitCallback(void *func_data, const double *x, const double *g,
                                 const double fx, const double xnorm, const double gnorm,
                                 const double step, int n, int k, int ls);

  public:
    typedef std::unique_ptr<PolyTrajOptimizer> Ptr;
  };

} // namespace ego_planner
#endif
