#pragma once

#include "TrajectoryOptComponents/PenaltyUtils.hpp"
#include "optimizer/traj_types.h"

namespace traj_opt_components
{
inline double accumulateEgoVelocityFeasibilityPenalty(const ego_planner::Vec3 &velocity,
                                                      const double max_velocity,
                                                      const double weight,
                                                      ego_planner::Vec3 &grad_velocity)
{
    if (weight <= 0.0)
    {
        return 0.0;
    }

    const double violation = velocity.squaredNorm() - max_velocity * max_velocity;
    double penalty = 0.0;
    double penalty_grad = 0.0;
    if (!positivePartCubic(violation, penalty, penalty_grad))
    {
        return 0.0;
    }

    grad_velocity += weight * penalty_grad * 2.0 * velocity;
    return weight * penalty;
}
} // namespace traj_opt_components
