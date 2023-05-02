#ifndef bamcp_threshold_solver
#define bamcp_threshold_solver
#include <string>
#include "state.h"
#include "multimodel_mdp.h"
#include "bamcp_solver.h"


/* Class to implement the Bayes adaptive Monte Carlo search approach to solving
Bayes adaptive MDPs which optimises the expected value if the reward is below
a threshold.

Attributes:
    maxTrials: the maximum number of trials to be used to find the next action.
    minReward: the threshold below which we should maximise the expected reward.
*/

class BAMCPThresholdSolver :  public BAMCPSolver {
private:
    float minReward;

public:
    //BAMCPThresholdSolver(){};
    BAMCPThresholdSolver(float minReward_, float biasMultiplier_=3.0) : BAMCPSolver(biasMultiplier_), minReward(minReward_){};
    virtual void updateNodes(
            std::vector<MCTSDecisionNode*>& dNodes,
            std::vector<MCTSChanceNode*>& cNodes,
            std::vector<float>& rewards);

    virtual float getReturn(
        std::vector<float>& rewards
    );
};

#endif
