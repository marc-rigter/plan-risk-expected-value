#ifndef cvar_decision_node
#define cvar_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "mcts_decision_node.h"

class CvarAdvDecisionNode;

/* class to be used for decision nodes in CVaR stochastic game MCTS. This
class inherits from the standard decision node but has the additional property
of having a continuous attribute alpha. Alpha means we are interested in
the average performance for the worst 100alpha% of runs: this is the same
notation as the paper Risk-Sensitive and Robust Decision making. Initially CVaR
starts below 1. It may be multiplied by a positive factor by the
adversary until it reaches 1. As soon as it reaches 1 it is the same as
optimising the expected value. */
class CvarDecisionNode : public MCTSDecisionNode {

protected:
    std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>> children;
    float alpha;

public:
    CvarDecisionNode(std::shared_ptr<MDP> pMDP_, const State state_, const float alpha_) : MCTSDecisionNode(pMDP_, state_), alpha(alpha_){};

    virtual std::shared_ptr<CvarAdvDecisionNode> createChild(std::string action);

    std::string expand();
    std::string getBestAction();
    std::string selectActionUCB(float biasFactor);
    void printPossiblePath();
    float getAlpha() {return alpha;};
    void setAlpha(float alpha_);
    std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>> getAdvChildren();
};

#endif
