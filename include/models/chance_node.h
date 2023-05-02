#ifndef chance_node
#define chance_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "decision_node.h"


class ChanceNode
{
protected:
    std::shared_ptr<MDP> pMDP;
    const State s;
    const std::string action;
    float localReward;

public:
    ChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action);
    float getLocalReward();
    std::string getAction();
    State getState();
};

#endif
