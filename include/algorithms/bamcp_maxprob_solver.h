#ifndef bamcp_maxprob_solver
#define bamcp_maxprob_solver
#include <string>
#include "state.h"
#include "multimodel_mdp.h"
#include "bamcp_solver.h"


/* Class to implement the Bayes adaptive Monte Carlo search approach to solving
Bayes adaptive MDPs, but maximising the probability that the reward exceeds
some threshold.

Attributes:
    maxTrials: the maximum number of trials to be used to find the next action.
    minReward: the reward threshold that we are trying to maximise staying
        above.
*/
class BAMCPMaxProbSolver :  public BAMCPSolver {
private:
    float minReward;

public:
    //BAMCPMaxProbSolver(){};
    BAMCPMaxProbSolver(float minReward_, float biasFactor_=3.0) : BAMCPSolver(biasFactor_), minReward(minReward_){};
    virtual void updateNodes(
            std::vector<MCTSDecisionNode*>& dNodes,
            std::vector<MCTSChanceNode*>& cNodes,
            std::vector<float>& rewards);

    virtual float getReturn(
        std::vector<float>& rewards
    );
};

#endif
