#ifndef _COST_FUNCTIONS_H_
#define _COST_FUNCTIONS_H_

#include "TrajectoryOptAdapters/TrackingIntegralCostAdapter.hpp"
#include "TrajectoryOptComponents/LinearTimeCost.hpp"

namespace ego_planner
{
  using TimeCostFunction = traj_opt_components::LinearTimeCost;
  using IntegralCostFunction = traj_opt_adapters::TrackingIntegralCostAdapter;
}

#endif
