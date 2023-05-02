#ifndef bamdp_rollout_policy
#define bamdp_rollout_policy
#include <string>
#include <iostream>
#include "state.h"
#include "belief.h"

class BamdpRolloutPolicy
{
private:

protected:

public:
    BamdpRolloutPolicy() {};
    virtual void updateRolloutPolicy(std::shared_ptr<Belief> b) = 0;
    virtual std::string getRolloutAction(State s, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief) = 0;
    virtual std::unordered_map<State, float, StateHash> getRolloutPerturbation(State s, std::string action, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief) = 0;
};

#endif
