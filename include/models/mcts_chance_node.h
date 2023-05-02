#ifndef mcts_chance_node
#define mcts_chance_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "mcts_decision_node.h"
#include "chance_node.h"

class MCTSDecisionNode;

/* Class to implement a chance node associated with a state action pair in
a search tree.

Attributes:
    observations: number of times this node has been visited in the search.
    currentValue: the current estimated state-action value of this node.
    children: a map from states to children. The children are of
        type ChanceNode.
    model: a reference to the MDP object which defines transitions.
    visited: list of successor states which have been visited so far.
    state: the state which this node is associated with.
    action: the action this node is associated with.
    localReward: the reward for executing the associated state action pair.
*/
class MCTSChanceNode : public ChanceNode
{
private:
    std::unordered_map<State, std::shared_ptr<MCTSDecisionNode>, StateHash> children;

protected:
    std::vector<State> visited;
    int observations;
    float totalReward;

    void visit();
public:
    MCTSChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action);
    float getCurrentValue();
    int getObservations();
    State selectRandomOutcome(std::shared_ptr<MDP> pSampledMDP = nullptr);
    std::unordered_map<State, std::shared_ptr<MCTSDecisionNode>, StateHash> getChildren();
    void updateNode(float returnValue);
};

#endif
