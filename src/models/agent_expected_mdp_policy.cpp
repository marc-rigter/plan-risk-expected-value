#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "mdp.h"
#include "agent_expected_mdp_policy.h"
#include "value_iteration.h"

AgentExpectedMDPPolicy::AgentExpectedMDPPolicy(std::shared_ptr<Belief> b) {
    std::cout << "Computing expected MDP expected value rollout policy..." << std::endl;
    // update expected MDP
    std::shared_ptr<MDP> expectedMDP = b->getExpectedMDP();

    // update value function for expected mdp
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;
    VI vi;
    std::tie(value, policy) = vi.valueIteration(*expectedMDP, true);
    expectedMDPValue = value;
    expectedMDPPolicy = policy;
    std::cout << "Done." << std::endl;
}

void AgentExpectedMDPPolicy::updateRolloutPolicy(std::shared_ptr<Belief> b){
    return;
}

std::string  AgentExpectedMDPPolicy::getRolloutAction(State s, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief){
    std::string action = expectedMDPPolicy[s];
    return action;
}

std::unordered_map<State, float, StateHash> AgentExpectedMDPPolicy::getRolloutPerturbation(State s, std::string action, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief){
    std::unordered_map<State, float, StateHash> transitionProbs = pCurrentBelief->getBeliefTransitionProbs(s, action);
    std::unordered_map<State, float, StateHash> perturbation;
    for(auto kv : transitionProbs){
        perturbation[kv.first] = 1.0;
    }
    return perturbation;
}
