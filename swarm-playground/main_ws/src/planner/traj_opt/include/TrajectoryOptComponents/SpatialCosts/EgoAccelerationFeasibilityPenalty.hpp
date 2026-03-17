#pragma once

#include "TrajectoryOptComponents/PenaltyUtils.hpp"
#include "optimizer/traj_types.h"

namespace traj_opt_components
{
inline double accumulateEgoAccelerationFeasibilityPenalty(const ego_planner::Vec3 &acceleration,
                                                          const double max_acceleration,
                                                          const double weight,
                                                          ego_planner::Vec3 &grad_acceleration)
{
    if (weight <= 0.0)
    {
        return 0.0;
    }

    const double violation = acceleration.squaredNorm() - max_acceleration * max_acceleration;
    double penalty = 0.0;
    double penalty_grad = 0.0;
    if (!positivePartCubic(violation, penalty, penalty_grad))
    {
        return 0.0;
    }

    grad_acceleration += weight * penalty_grad * 2.0 * acceleration;
    return weight * penalty;
}
} // namespace traj_opt_components
