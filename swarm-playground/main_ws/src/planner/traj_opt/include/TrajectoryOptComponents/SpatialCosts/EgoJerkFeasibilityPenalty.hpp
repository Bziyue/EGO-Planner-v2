#pragma once

#include "TrajectoryOptComponents/PenaltyUtils.hpp"
#include "optimizer/traj_types.h"

namespace traj_opt_components
{
inline double accumulateEgoJerkFeasibilityPenalty(const ego_planner::Vec3 &jerk,
                                                  const double max_jerk,
                                                  const double weight,
                                                  ego_planner::Vec3 &grad_jerk)
{
    if (weight <= 0.0)
    {
        return 0.0;
    }

    const double violation = jerk.squaredNorm() - max_jerk * max_jerk;
    double penalty = 0.0;
    double penalty_grad = 0.0;
    if (!positivePartCubic(violation, penalty, penalty_grad))
    {
        return 0.0;
    }

    grad_jerk += weight * penalty_grad * 2.0 * jerk;
    return weight * penalty;
}
} // namespace traj_opt_components
