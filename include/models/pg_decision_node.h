#ifndef pg_decision_node
#define pg_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "decision_node.h"
#include "pg_chance_node.h"

class PGChanceNode;

/* Decision node class for policy gradient method.
*/
class PGDecisionNode : public DecisionNode
{
protected:
    std::unordered_map<std::string, float> actionWeights;
    std::unordered_map<std::string, std::shared_ptr<PGChanceNode>> children;
    std::vector<std::string> actionsTaken;

public:
    PGDecisionNode(std::shared_ptr<MDP> pMDP_, const State state_);
    std::string selectActionSoftmax();
    std::unordered_map<std::string, std::shared_ptr<PGChanceNode>> getChildren();
    std::shared_ptr<PGChanceNode> createChild(std::string action);
    std::unordered_map<std::string, float> getActionWeights();
    void setActionWeight(std::string action, float value);
};


#endif
