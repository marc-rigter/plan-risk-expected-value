#ifndef multimodel_mdp
#define multimodel_mdp
#include <string>
#include "mdp.h"
#include <unordered_map>


/* Implements a multi-model MDP with different probability weightings
associated with each of the MDPs. Note that it is assumed that all of the MDP
samples have the same state and action spaces and reward functions, but
may have different transition functions.

Attributes:
    weightings: a mapping from shared pointers to MDPs to probabilties of the
        corresponding MDP.
*/
class MultiModelMDP
{
private:
    std::unordered_map<std::shared_ptr<MDP>, float> weights;

public:
    MultiModelMDP() {};
    MultiModelMDP(std::unordered_map<std::shared_ptr<MDP>, float> weights_);
    std::shared_ptr<MDP> sampleModel();
    void updateBelief(State initState, std::string action, State nextState);
    std::unordered_map<std::shared_ptr<MDP>, float> getWeights();
};

#endif
