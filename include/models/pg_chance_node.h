#ifndef pg_chance_node
#define pg_chance_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "decision_node.h"
#include "chance_node.h"
#include "pg_decision_node.h"

class PGDecisionNode;

/* Class to implement a chance node associated with a state action pair in
a search tre for policy gradient method.
*/
class PGChanceNode : public ChanceNode
{
protected:
    std::vector<State> visited;
    std::unordered_map<State, std::shared_ptr<PGDecisionNode>, StateHash> children;

public:
    PGChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action);
    State sampleSuccessor(std::shared_ptr<MDP> pSampledMDP);
    std::unordered_map<State, std::shared_ptr<PGDecisionNode>, StateHash> getChildren();
};

#endif
