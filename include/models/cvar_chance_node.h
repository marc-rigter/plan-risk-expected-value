#ifndef cvar_chance_node
#define cvar_chance_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "mcts_chance_node.h"
#include "cvar_decision_node.h"

class CvarChanceNode : public MCTSChanceNode {
public:

    CvarChanceNode(std::shared_ptr<MDP> pMDP_, const State s_, float alpha_, const std::string action_,
        std::unordered_map<State, float, StateHash> advPerturbation_) :
    MCTSChanceNode(pMDP_, s_, action_), alpha(alpha_), advPerturbation(advPerturbation_) {};

    virtual State sampleSuccessor();
    virtual State sampleUnperturbedSuccessor();
    virtual std::shared_ptr<CvarDecisionNode> createChild(State nextState);

    State sampleSuccessorAddChild();
    void addChild(State nextState);

    std::unordered_map<State, float, StateHash> getPerturbation();
    std::unordered_map<State, std::shared_ptr<CvarDecisionNode>, StateHash> getCvarChildren();
    float getAlpha();

protected:
    float alpha;
    std::unordered_map<State, float, StateHash> advPerturbation;
    std::unordered_map<State, std::shared_ptr<CvarDecisionNode>, StateHash> cvarChildren;
};

#endif
