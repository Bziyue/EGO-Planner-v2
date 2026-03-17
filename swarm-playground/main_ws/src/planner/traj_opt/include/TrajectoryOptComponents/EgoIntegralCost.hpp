#pragma once

#include "optimizer/traj_types.h"

#include <plan_env/grid_map.h>

namespace traj_opt_components
{
class EgoIntegralCost
{
public:
  GridMap::Ptr grid_map;
  ego_planner::ConstraintPoints *cps;
  SwarmTrajData *swarm_trajs;
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

    cost += obstacleGradCostP(cp_idx, p, gp);
    cost += swarmGradCostP(cp_idx, t_global, p, v, gp, gt);
    cost += feasibilityGradCost(v, a, j, gv, ga, gj);

    ++step_in_seg_;
    return cost;
  }

private:
  double obstacleGradCostP(int cp_idx, const ego_planner::Vec3 &p, ego_planner::Vec3 &gradp) const
  {
    if (!cps || cp_idx == 0 || cp_idx >= cps->cp_size ||
        cp_idx > ego_planner::ConstraintPoints::two_thirds_id(cps->points, touch_goal))
      return 0.0;

    double costp = 0.0;
    for (size_t k = 0; k < cps->direction[cp_idx].size(); ++k)
    {
      ego_planner::Vec3 ray = (p - cps->base_point[cp_idx][k]);
      double dist = ray.dot(cps->direction[cp_idx][k]);
      double dist_err = obs_clearance - dist;
      double dist_err_soft = obs_clearance_soft - dist;
      ego_planner::Vec3 dist_grad = cps->direction[cp_idx][k];

      if (dist_err > 0)
      {
        costp += wei_obs * pow(dist_err, 3);
        gradp += -wei_obs * 3.0 * dist_err * dist_err * dist_grad;
      }

      if (dist_err_soft > 0)
      {
        const double r = 0.05;
        const double rsqr = r * r;
        const double term = sqrt(1.0 + dist_err_soft * dist_err_soft / rsqr);
        costp += wei_obs_soft * rsqr * (term - 1.0);
        gradp += -wei_obs_soft * dist_err_soft / term * dist_grad;
      }
    }
    accumulated_costs(0) += costp;
    return costp;
  }

  double swarmGradCostP(int cp_idx, double t_global, const ego_planner::Vec3 &p, const ego_planner::Vec3 &v,
                        ego_planner::Vec3 &gradp, double &gt) const
  {
    if (!swarm_trajs || !cps || cp_idx <= 0 || cp_idx >= cps->cp_size ||
        cp_idx > ego_planner::ConstraintPoints::two_thirds_id(cps->points, touch_goal))
      return 0.0;

    double costp = 0.0;
    constexpr double a_param = 2.0, b_param = 1.0;
    constexpr double inv_a2 = 1.0 / (a_param * a_param), inv_b2 = 1.0 / (b_param * b_param);

    for (size_t id = 0; id < swarm_trajs->size(); ++id)
    {
      if ((swarm_trajs->at(id).drone_id < 0) || swarm_trajs->at(id).drone_id == drone_id)
        continue;

      const double traj_i_start_time = swarm_trajs->at(id).start_time;
      const double pt_time = (t_now - traj_i_start_time) + t_global;
      const double clearance = (swarm_clearance + swarm_trajs->at(id).des_clearance) * 1.5;
      const double clearance2 = clearance * clearance;

      ego_planner::Vec3 swarm_p, swarm_v;
      if (pt_time < swarm_trajs->at(id).duration)
      {
        swarm_p = swarm_trajs->at(id).traj.evaluate(swarm_trajs->at(id).traj.getStartTime() + pt_time, SplineTrajectory::Deriv::Pos);
        swarm_v = swarm_trajs->at(id).traj.evaluate(swarm_trajs->at(id).traj.getStartTime() + pt_time, SplineTrajectory::Deriv::Vel);
      }
      else
      {
        const double end_t = swarm_trajs->at(id).traj.getStartTime() + swarm_trajs->at(id).duration;
        swarm_v = swarm_trajs->at(id).traj.evaluate(end_t, SplineTrajectory::Deriv::Vel);
        swarm_p = swarm_trajs->at(id).traj.evaluate(end_t, SplineTrajectory::Deriv::Pos) +
                  (pt_time - swarm_trajs->at(id).duration) * swarm_v;
      }
      const ego_planner::Vec3 dist_vec = p - swarm_p;
      const double ellip_dist2 = dist_vec(2) * dist_vec(2) * inv_a2 +
                                 (dist_vec(0) * dist_vec(0) + dist_vec(1) * dist_vec(1)) * inv_b2;
      const double dist2_err = clearance2 - ellip_dist2;
      const double dist2_err2 = dist2_err * dist2_err;
      const double dist2_err3 = dist2_err2 * dist2_err;

      if (dist2_err3 > 0)
      {
        costp += wei_swarm * dist2_err3;
        ego_planner::Vec3 dJ_dP = wei_swarm * 3 * dist2_err2 * (-2) *
                                  ego_planner::Vec3(inv_b2 * dist_vec(0), inv_b2 * dist_vec(1), inv_a2 * dist_vec(2));
        gradp += dJ_dP;
        gt += dJ_dP.dot(-swarm_v);
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

  double feasibilityGradCost(const ego_planner::Vec3 &v, const ego_planner::Vec3 &a, const ego_planner::Vec3 &j,
                             ego_planner::Vec3 &gv, ego_planner::Vec3 &ga, ego_planner::Vec3 &gj) const
  {
    double cost = 0.0;

    double vpen = v.squaredNorm() - max_vel * max_vel;
    if (vpen > 0)
    {
      gv += wei_feas * 6 * vpen * vpen * v;
      cost += wei_feas * vpen * vpen * vpen;
    }

    double apen = a.squaredNorm() - max_acc * max_acc;
    if (apen > 0)
    {
      ga += wei_feas * 6 * apen * apen * a;
      cost += wei_feas * apen * apen * apen;
    }

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
}
