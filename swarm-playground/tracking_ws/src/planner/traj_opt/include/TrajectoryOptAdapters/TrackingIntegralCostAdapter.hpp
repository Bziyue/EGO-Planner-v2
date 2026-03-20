#pragma once

#include "TrajectoryOptAdapters/EgoIntegralCostAdapter.hpp"

namespace traj_opt_adapters
{
class TrackingIntegralCostAdapter : public EgoIntegralCostAdapter
{
public:
  using Types = EgoPlanningTypesAdapter;

  double wei_tracking{0.0};
  Types::Vec3 relative_tracking_p{Types::Vec3::Constant(-10000.0)};
  Types::Vec3 object_p{Types::Vec3::Constant(-10000.0)};
  Types::Vec3 object_v{Types::Vec3::Constant(-10000.0)};
  Eigen::Quaterniond object_q{1.0, 0.0, 0.0, 0.0};
  bool start_tracking{false};

  double operator()(double t, double t_global, int seg_idx, int step_in_seg,
                    const Types::Vec3 &p, const Types::Vec3 &v,
                    const Types::Vec3 &a, const Types::Vec3 &j, const Types::Vec3 &s,
                    Types::Vec3 &gp, Types::Vec3 &gv, Types::Vec3 &ga,
                    Types::Vec3 &gj, Types::Vec3 &gs, double &gt) const
  {
    const double base_cost = EgoIntegralCostAdapter::operator()(t, t_global, seg_idx, step_in_seg, p, v, a, j, s, gp, gv, ga, gj, gs, gt);
    if (!start_tracking || !cps || seg_idx < 0)
    {
      return base_cost;
    }

    const int cp_idx = seg_idx * cps_per_piece + step_in_seg;
    if (cp_idx <= 0 || cp_idx > Types::ConstraintPoints::two_thirds_id(cps->points, touch_goal))
    {
      return base_cost;
    }
    if (object_p.x() < -9999.0 || relative_tracking_p.x() < -9999.0)
    {
      return base_cost;
    }

    const Types::Vec3 object_p_t = object_p + object_v * t + object_q.matrix() * relative_tracking_p;
    const Types::Vec3 dJ_dp = 2.0 * (p - object_p_t);
    const double tracking_cost = wei_tracking * 0.25 * dJ_dp.squaredNorm();
    gp += wei_tracking * dJ_dp;
    gt += wei_tracking * dJ_dp.dot(v - object_v);
    return base_cost + tracking_cost;
  }
};
} // namespace traj_opt_adapters
