#ifndef mdp_example
#define mdp_example
#include "mdp.h"
#include "fully_tied_dirichlet_belief.h"
#include "tied_dirichlet_belief.h"

std::shared_ptr<MDP> makeSimpleMDP(int xGoal, int yGoal, float penalty);
std::shared_ptr<MDP> makeIndoorNavMDP(float wideSuccess, float narrowSuccess);
std::shared_ptr<MDP> makeBettingMDP(float successProb, int stages, const std::vector<int>& bets = std::vector<int>());
std::shared_ptr<MDP> makeMedicalMDP(int days, int seed);
std::shared_ptr<FullyTiedDirichletBelief> getBettingGameBelief(float winCounts, float loseCounts, int stages, std::shared_ptr<MDP> pMDP);
std::shared_ptr<MDP> marsRoverMDP(int horizon);
std::vector<std::vector<std::string>> getRoverMatrix();
std::shared_ptr<Belief> getMarsRoverBelief(std::shared_ptr<MDP> marsRoverMDP, int horizon, float initCount, bool fullyTied);
std::shared_ptr<MDP> trafficMDP(int horizon);
std::shared_ptr<MDP> trafficMDPRefactored(int horizon);
std::shared_ptr<TiedDirichletBelief> getTrafficBelief(std::shared_ptr<MDP> trafficMDP, int horizon, std::vector<float> initCounts, bool refactored=false);
std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> getTrafficOutcomes();
std::shared_ptr<MDP> makeInventoryMDP(int stages, int maxInventory);
std::shared_ptr<MDP> makeInventoryRandomWalkMDP(int stages, int maxInventory);

#endif
