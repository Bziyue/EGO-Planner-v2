#pragma once

#include "TrajectoryOptComponents/PenaltyUtils.hpp"
#include "optimizer/traj_types.h"

namespace traj_opt_components
{
inline double accumulateEgoObstacleHardPenalty(const double distance,
                                               const double clearance,
                                               const double weight,
                                               const ego_planner::Vec3 &direction,
                                               ego_planner::Vec3 &grad_position)
{
    if (weight <= 0.0)
    {
        return 0.0;
    }

    const double violation = clearance - distance;
    double penalty = 0.0;
    double penalty_grad = 0.0;
    if (!positivePartCubic(violation, penalty, penalty_grad))
    {
        return 0.0;
    }

    grad_position += -weight * penalty_grad * direction;
    return weight * penalty;
}
} // namespace traj_opt_components
