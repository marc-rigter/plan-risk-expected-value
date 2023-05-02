#ifndef hardcoded_solver
#define  hardcoded_solver
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "bamdp_solver.h"

/* Class to implement a hardcoded stationary policy to apply to a Bayes
adaptive MDP.

Attributes:
    policy: the hardcoded stationary policy.
*/
class HardcodedSolver :  public BAMDPSolver {
private:
    std::unordered_map<State, std::string, StateHash> policy;

public:
    HardcodedSolver(std::unordered_map<State, std::string, StateHash> policy_) : policy(policy_) {};
    virtual std::string getNextAction(MultiModelMDP& mmdp, State currentState, int horizon){
        return policy[currentState];
    };
};

#endif
