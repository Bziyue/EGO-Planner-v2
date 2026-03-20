#pragma once

#include "TrajectoryOptAdapters/EgoIntegralCostAdapter.hpp"

namespace traj_opt_adapters
{
class FormationIntegralCostAdapter : public EgoIntegralCostAdapter
{
public:
  using Types = EgoPlanningTypesAdapter;

  double wei_formation{0.0};
  int formation_num{-1};
  Eigen::MatrixXd formation;
  Types::Vec3 formation_start{Types::Vec3::Zero()};
  Types::Vec3 formation_end{Types::Vec3::Zero()};

  double operator()(double t, double t_global, int seg_idx, int step_in_seg,
                    const Types::Vec3 &p, const Types::Vec3 &v,
                    const Types::Vec3 &a, const Types::Vec3 &j, const Types::Vec3 &s,
                    Types::Vec3 &gp, Types::Vec3 &gv, Types::Vec3 &ga,
                    Types::Vec3 &gj, Types::Vec3 &gs, double &gt) const
  {
    const double base_cost = EgoIntegralCostAdapter::operator()(t, t_global, seg_idx, step_in_seg, p, v, a, j, s, gp, gv, ga, gj, gs, gt);
    if (!cps || seg_idx < 0)
    {
      return base_cost;
    }

    const int cp_idx = seg_idx * cps_per_piece + step_in_seg;
    if (cp_idx <= 0 || cp_idx > Types::ConstraintPoints::two_thirds_id(cps->points, touch_goal))
    {
      return base_cost;
    }
    if (formation_num <= 1 || formation.cols() < formation_num)
    {
      return base_cost;
    }
    if (!swarm_trajs || ((int)swarm_trajs->size() < formation_num && drone_id != formation_num - 1))
    {
      return base_cost;
    }

    const Types::Vec3 direction = formation_end - formation_start;
    if (direction.squaredNorm() < 1e-12)
    {
      return base_cost;
    }

    const Types::Vec3 axis = direction.normalized();
    const double pt_time = t_now + t;

    double l = 0.0;
    double dl_dt = 0.0;
    const int id_end = drone_id == (formation_num - 1) ? formation_num - 1 : formation_num;
    for (int id = 0; id < id_end; ++id)
    {
      if (id >= (int)swarm_trajs->size())
      {
        break;
      }
      const auto &swarm_traj = swarm_trajs->at(id);
      if (swarm_traj.drone_id < 0 || swarm_traj.drone_id == drone_id)
      {
        continue;
      }

      Types::Vec3 swarm_p;
      Types::Vec3 swarm_v;
      const double traj_start_time = swarm_traj.start_time;
      if (pt_time < traj_start_time + swarm_traj.duration)
      {
        swarm_p = swarm_traj.traj.evaluate(pt_time - traj_start_time + swarm_traj.traj.getStartTime(), SplineTrajectory::Deriv::Pos);
        swarm_v = swarm_traj.traj.evaluate(pt_time - traj_start_time + swarm_traj.traj.getStartTime(), SplineTrajectory::Deriv::Vel);
      }
      else
      {
        const double exceed_time = pt_time - (traj_start_time + swarm_traj.duration);
        const double traj_end = swarm_traj.traj.getEndTime();
        swarm_v = swarm_traj.traj.evaluate(traj_end, SplineTrajectory::Deriv::Vel);
        swarm_p = swarm_traj.traj.evaluate(traj_end, SplineTrajectory::Deriv::Pos) + exceed_time * swarm_v;
      }

      l += (swarm_p - formation_start).dot(axis) - formation(0, id);
      dl_dt += axis.dot(swarm_v);
    }

    l /= (formation_num - 1);
    dl_dt /= (formation_num - 1);

    Types::Vec3 target;
    target.x() = (axis.x() * (formation(0, drone_id) + l) - axis.y() * formation(1, drone_id)) + formation_start.x();
    target.y() = (axis.y() * (formation(0, drone_id) + l) + axis.x() * formation(1, drone_id)) + formation_start.y();
    target.z() = axis.z() * l + formation(2, drone_id) + formation_start.z();

    const Types::Vec3 dJ_dp = 2.0 * (p - target);
    const double formation_cost = wei_formation * 0.25 * dJ_dp.squaredNorm();
    gp += wei_formation * dJ_dp;
    gt += wei_formation * dJ_dp.dot(v - axis * dl_dt);
    return base_cost + formation_cost;
  }
};
} // namespace traj_opt_adapters
