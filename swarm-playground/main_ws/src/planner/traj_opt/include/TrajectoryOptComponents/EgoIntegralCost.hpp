#pragma once

#include "TrajectoryOptComponents/SpatialCosts/EgoFeasibilityPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoObstaclePenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoSwarmPenalty.hpp"
#include "optimizer/traj_types.h"

#include <plan_env/grid_map.h>

namespace traj_opt_components
{
class EgoIntegralCost
{
public:
  GridMap::Ptr grid_map;
  ego_planner::ConstraintPoints *cps;
  ego_planner::SwarmTrajData *swarm_trajs;
  double wei_obs, wei_obs_soft, wei_swarm, wei_feas, wei_sqrvar;
  double obs_clearance, obs_clearance_soft, swarm_clearance;
  double max_vel, max_acc, max_jer;
  int drone_id;
  double t_now;
  bool touch_goal;
  int cps_per_piece;
  mutable int current_seg_;
  mutable int step_in_seg_;
  mutable std::vector<double> *min_ellip_dist2_ptr;
  mutable Eigen::VectorXd accumulated_costs;
  mutable std::vector<double> segment_dt_;

  EgoIntegralCost()
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
                    const ego_planner::Vec3 &p, const ego_planner::Vec3 &v,
                    const ego_planner::Vec3 &a, const ego_planner::Vec3 &j, const ego_planner::Vec3 &s,
                    ego_planner::Vec3 &gp, ego_planner::Vec3 &gv, ego_planner::Vec3 &ga,
                    ego_planner::Vec3 &gj, ego_planner::Vec3 &gs, double &gt) const
  {
    double cost = 0.0;

    if (seg_idx != current_seg_)
    {
      current_seg_ = seg_idx;
      step_in_seg_ = 0;
    }

    const int cp_idx = seg_idx * cps_per_piece + step_in_seg_;

    if (cps && cp_idx < cps->cp_size)
      cps->points.col(cp_idx) = p;

    const double obstacle_cost = accumulateEgoObstaclePenalty(cp_idx,
                                                              cps,
                                                              touch_goal,
                                                              p,
                                                              obs_clearance,
                                                              obs_clearance_soft,
                                                              wei_obs,
                                                              wei_obs_soft,
                                                              gp);
    const double swarm_cost = accumulateEgoSwarmPenalty(cp_idx,
                                                        cps,
                                                        touch_goal,
                                                        swarm_trajs,
                                                        drone_id,
                                                        t_now,
                                                        t_global,
                                                        swarm_clearance,
                                                        wei_swarm,
                                                        p,
                                                        gp,
                                                        gt,
                                                        min_ellip_dist2_ptr);
    const double feasibility_cost = accumulateEgoFeasibilityPenalty(v,
                                                                    a,
                                                                    j,
                                                                    max_vel,
                                                                    max_acc,
                                                                    max_jer,
                                                                    wei_feas,
                                                                    gv,
                                                                    ga,
                                                                    gj);

    accumulated_costs(0) += obstacle_cost;
    accumulated_costs(1) += swarm_cost;
    accumulated_costs(2) += feasibility_cost;
    cost += obstacle_cost + swarm_cost + feasibility_cost;

    ++step_in_seg_;
    return cost;
  }
};
}
