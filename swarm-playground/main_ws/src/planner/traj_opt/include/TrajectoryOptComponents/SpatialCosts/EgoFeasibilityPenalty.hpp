#pragma once

#include "TrajectoryOptComponents/SpatialCosts/EgoAccelerationFeasibilityPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoJerkFeasibilityPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/EgoVelocityFeasibilityPenalty.hpp"
#include "optimizer/traj_types.h"

namespace traj_opt_components
{
inline double accumulateEgoFeasibilityPenalty(const ego_planner::Vec3 &velocity,
                                              const ego_planner::Vec3 &acceleration,
                                              const ego_planner::Vec3 &jerk,
                                              const double max_vel,
                                              const double max_acc,
                                              const double max_jer,
                                              const double weight,
                                              ego_planner::Vec3 &grad_velocity,
                                              ego_planner::Vec3 &grad_acceleration,
                                              ego_planner::Vec3 &grad_jerk)
{
    if (weight <= 0.0)
    {
        return 0.0;
    }

    return accumulateEgoVelocityFeasibilityPenalty(velocity, max_vel, weight, grad_velocity) +
           accumulateEgoAccelerationFeasibilityPenalty(acceleration, max_acc, weight, grad_acceleration) +
           accumulateEgoJerkFeasibilityPenalty(jerk, max_jer, weight, grad_jerk);
}
} // namespace traj_opt_components
