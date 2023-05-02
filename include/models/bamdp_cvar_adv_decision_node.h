#ifndef bamdp_cvar_adv_decision_node
#define bamdp_cvar_adv_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "cvar_adv_decision_node.h"
#include "belief.h"


/* class to be used for decision nodes of the adversary in CVaR
MDP optimisation. */
class BamdpCvarAdvDecisionNode : public CvarAdvDecisionNode {

public:
    BamdpCvarAdvDecisionNode(
            std::shared_ptr<MDP> pMDP_,
            std::shared_ptr<Belief> b_,
            const State state_,
            const float alpha_,
            const std::string action_) : CvarAdvDecisionNode(pMDP_, state_, alpha_, action_), b(b_)
    {
    };

    std::unordered_map<State, float, StateHash> getSuccessorProbs();
    std::shared_ptr<CvarChanceNode> createChild(std::unordered_map<State, float, StateHash> perturbationMapping);
    int expandHeuristicPerturbation(std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy, int stepsRemaining);

protected:
    std::shared_ptr<Belief> b;
    bool successorProbsComputed = false;
    std::unordered_map<State, float, StateHash> successorProbs;
};

#endif
