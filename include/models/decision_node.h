#ifndef decision_node
#define decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "chance_node.h"

class DecisionNode
{
protected:
    std::shared_ptr<MDP> pMDP;
    State s;
    std::vector<std::string> possibleActions;

public:
    DecisionNode(std::shared_ptr<MDP> pMDP_, const State state_);
    DecisionNode();
    State getState();
};


#endif
