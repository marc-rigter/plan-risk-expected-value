#ifndef mcts_decision_node
#define mcts_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "mcts_chance_node.h"
#include "decision_node.h"

class MCTSChanceNode;

/* Decision node class for Monte Carlo tree search.

Attributes:
    observations: number of times this node has been visited in the search.
    currentValue: the current estimated value of this node.
    children: a map from actions (strings) to children. The children are of
        type ChanceNode.
    model: a reference to the MDP object which defines transitions.
    possibleActions: list of possible actions at this state.
    notTakenActions: list of actions which have not been tried at this node.
    state: the state which this node is associated with.
*/
class MCTSDecisionNode : public DecisionNode
{
private:
    std::unordered_map<std::string, std::shared_ptr<MCTSChanceNode>> children;

protected:
    std::vector<std::string> notTakenActions;
    int observations;
    float totalReward;

    void visit();

public:
    MCTSDecisionNode(std::shared_ptr<MDP> pMDP_, const State state_);
    MCTSDecisionNode();
    virtual std::string selectActionUCB(float biasFactor=3.0f);
    virtual std::string expand();
    int getObservations();
    std::vector<std::string> getNotTakenActions();
    float getCurrentValue();
    virtual std::string getBestAction();
    std::unordered_map<std::string, std::shared_ptr<MCTSChanceNode>> getChildren();
    void updateNode(float returnValue);
    bool isFullyExpanded();
    bool isExpanded();
};


#endif
