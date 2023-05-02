#ifndef cvar_expected_mdp_policy
#define cvar_expected_mdp_policy
#include <string>
#include <iostream>
#include "state.h"
#include "belief.h"
#include "bamdp_rollout_policy.h"
#include "cvar_value_iteration.h"

class CvarExpectedMDPPolicy : public BamdpRolloutPolicy
{
private:

protected:
    std::shared_ptr<CvarValueIteration> pSolver;
    std::shared_ptr<MDP> expectedMDP;
    GRBEnv env = GRBEnv();
    std::unordered_map<State, std::map<std::string, std::string>, StateHash> cvarPolicy;
    std::unordered_map<State, std::map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> cvarPerturbationPolicy;

public:
    CvarExpectedMDPPolicy(std::shared_ptr<Belief> b);
    void updateRolloutPolicy(std::shared_ptr<Belief> b);
    std::string getRolloutAction(State s, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief);
    std::unordered_map<State, float, StateHash> getRolloutPerturbation(State s, std::string action, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief);
};

#endif
