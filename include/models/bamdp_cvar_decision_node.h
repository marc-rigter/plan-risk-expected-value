#ifndef bamdp_cvar_decision_node
#define bamdp_cvar_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "cvar_decision_node.h"
#include "belief.h"

class CvarAdvDecisionNode;

class BamdpCvarDecisionNode : public CvarDecisionNode {
public:
    BamdpCvarDecisionNode(
            std::shared_ptr<MDP> pMDP_,
            std::shared_ptr<Belief> b_,
            const State state_,
            const float alpha_)
    : CvarDecisionNode(pMDP_, state_, alpha_), b(b_) {};

    std::shared_ptr<CvarAdvDecisionNode> createChild(std::string action);
    std::shared_ptr<Belief> getBelief();

protected:
    std::shared_ptr<Belief> b;
};

#endif
