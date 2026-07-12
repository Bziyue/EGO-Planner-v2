#pragma once

#include "SplineTrajectory/IntegralPointInfo.hpp"
#include "TrajectoryOptAdapters/EgoPlanningTypesAdapter.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoFeasibilityPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoObstaclePenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoSwarmPenalty.hpp"

#include <vector>

namespace traj_opt_adapters
{
class EgoIntegralCostAdapter
{
public:
  using Types = EgoPlanningTypesAdapter;

  Types::GridMapPtr grid_map;
  Types::ConstraintPoints *cps;
  Types::SwarmTrajData *swarm_trajs;
  double wei_obs, wei_obs_soft, wei_swarm, wei_feas, wei_sqrvar;
  double obs_clearance, obs_clearance_soft, swarm_clearance;
  double max_vel, max_acc, max_jer;
  int drone_id;
  double t_now;
  bool touch_goal;
  mutable std::vector<double> *min_ellip_dist2_ptr;
  mutable Eigen::VectorXd accumulated_costs;

  EgoIntegralCostAdapter()
      : grid_map(nullptr), cps(nullptr), swarm_trajs(nullptr),
        wei_obs(0), wei_obs_soft(0), wei_swarm(0), wei_feas(0), wei_sqrvar(0),
        obs_clearance(0), obs_clearance_soft(0), swarm_clearance(0),
        max_vel(0), max_acc(0), max_jer(0),
        drone_id(-1), t_now(0), touch_goal(false),
        min_ellip_dist2_ptr(nullptr)
  {
    accumulated_costs.resize(4);
    accumulated_costs.setZero();
  }

  void beginEvaluation() const
  {
    accumulated_costs.setZero();
  }

  double operator()(const SplineTrajectory::IntegralPointInfo &point,
                    const Types::Vec3 &p, const Types::Vec3 &v,
                    const Types::Vec3 &a, const Types::Vec3 &j, const Types::Vec3 &s,
                    Types::Vec3 &gp, Types::Vec3 &gv, Types::Vec3 &ga,
                    Types::Vec3 &gj, Types::Vec3 &gs, double &gt) const
  {
    double cost = 0.0;
    (void)s;
    (void)gs;

    const int cp_idx = point.segment_index * point.step_count + point.step_index;

    if (cps && cp_idx < cps->cp_size)
      cps->points.col(cp_idx) = p;

    const double obstacle_cost = traj_opt_components::accumulateEgoObstaclePenalty(cp_idx,
                                                                                    cps,
                                                                                    touch_goal,
                                                                                    p,
                                                                                    obs_clearance,
                                                                                    obs_clearance_soft,
                                                                                    wei_obs,
                                                                                    wei_obs_soft,
                                                                                    gp);
    const double swarm_cost = traj_opt_components::accumulateEgoSwarmPenalty(cp_idx,
                                                                              cps,
                                                                              touch_goal,
                                                                              swarm_trajs,
                                                                              drone_id,
                                                                              t_now,
                                                                              point.global_time,
                                                                              swarm_clearance,
                                                                              wei_swarm,
                                                                              p,
                                                                              gp,
                                                                              gt,
                                                                              min_ellip_dist2_ptr);
    const double feasibility_cost = traj_opt_components::accumulateEgoFeasibilityPenalty(v,
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

    return cost;
  }
};
} // namespace traj_opt_adapters
