#ifndef bamdp_solver
#define bamdp_solver
#include <string>
#include "state.h"
#include "multimodel_mdp.h"
#include "hist.h"


/* Base class for Bayes adaptive MDP solvers. */
class BAMDPSolver{
protected:

    // variable to keep track of the cost incurred thus far in the episode
    float rewardThisEpisode = 0.0;

public:
    virtual std::string getNextAction(MultiModelMDP& mmdp, State currentState, int horizon) = 0;
    virtual Hist executeEpisode(
                MultiModelMDP& mmdp,
                std::shared_ptr<MDP> pTrueModel,
                State currentState,
                int horizon);
    void resetEpisodeReward();
    void updateEpisodeReward(float reward);

};

#endif
