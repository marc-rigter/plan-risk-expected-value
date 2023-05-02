#ifndef agent_expected_mdp_policy
#define agent_expected_mdp_policy
#include <string>
#include <iostream>
#include "state.h"
#include "belief.h"
#include "bamdp_rollout_policy.h"

class AgentExpectedMDPPolicy : public BamdpRolloutPolicy
{
private:

protected:
    std::unordered_map<State, float, StateHash> expectedMDPValue;
    std::unordered_map<State, std::string, StateHash> expectedMDPPolicy;

public:
    AgentExpectedMDPPolicy(std::shared_ptr<Belief> b);
    void updateRolloutPolicy(std::shared_ptr<Belief> b);
    std::string getRolloutAction(State s, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief);
    std::unordered_map<State, float, StateHash> getRolloutPerturbation(State s, std::string action, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief);
};

#endif
