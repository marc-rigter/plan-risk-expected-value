#ifndef bamdp_cvar_chance_node
#define bamdp_cvar_chance_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "cvar_chance_node.h"
#include "cvar_decision_node.h"
#include "belief.h"


class BamdpCvarChanceNode : public CvarChanceNode {
public:
    BamdpCvarChanceNode(
            std::shared_ptr<MDP> pMDP_,
            std::shared_ptr<Belief> b_,
            const State s_,
            float alpha_,
            const std::string action_,
            std::unordered_map<State, float, StateHash> advPerturbation_,
            std::unordered_map<State, float, StateHash> nominalTransProbs_)
            :
            CvarChanceNode(pMDP_, s_, alpha_, action_, advPerturbation_),
            b(b_),
            nominalTransProbs(nominalTransProbs_)
            {};

    State sampleSuccessor();
    State sampleUnperturbedSuccessor();
    std::shared_ptr<CvarDecisionNode> createChild(State nextState);

protected:
    std::shared_ptr<Belief> b;
    std::shared_ptr<Belief> getNewBelief(State s, std::string action, State nextState);
    std::unordered_map<State, float, StateHash> nominalTransProbs;
};

#endif
