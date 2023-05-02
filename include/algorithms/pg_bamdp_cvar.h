#ifndef cvar_pg
#define cvar_pg
#include <string>
#include "state.h"
#include "pg_decision_node.h"
#include "belief.h"
#include "hist.h"

/* Class to implement the policy gradient approach to optimise
*/
class BamdpCvarPG {
protected:
    float learningRate;
    std::shared_ptr<PGDecisionNode> rootNode;
    std::shared_ptr<Belief> rootBelief;
    State initState;
    float alpha;
    int horizon;
    std::vector<std::shared_ptr<MDP>> rootSamples;


public:

    BamdpCvarPG(
            float learningRate_,
            std::shared_ptr<Belief> pBelief,
            State initState_,
            float alpha_,
            int horizon_
    );

    void gradientUpdate(
            std::vector<std::vector<std::shared_ptr<PGDecisionNode>>> trajectories,
            std::vector<std::vector<std::string>> trajectoryActions,
            std::vector<float> returns,
            float varEstimate
    );

    virtual void runTrials(
        int batchSize,
        int iterations
    );

    virtual Hist executeEpisode(std::shared_ptr<MDP> pTrueModel);
};

#endif
