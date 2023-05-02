#ifndef mcts_cvar_sg_offline
#define mcts_cvar_sg_offline
#include <string>
#include "state.h"
#include "mcts.h"
#include "cvar_chance_node.h"
#include "cvar_decision_node.h"
#include "cvar_game_hist.h"
#include "bamdp_rollout_policy.h"
#include "mcts_cvar_sg.h"

/* Class to implement monte carlo tree search to solve game optimising CVaR
in MDPs
*/
class CvarMCTSOffline : public CvarMCTS {
public:
    CvarMCTSOffline(
        std::shared_ptr<MDP> pMDP_,
        std::shared_ptr<CvarDecisionNode> rootNode_,
        int horizon_,
        std::string optim="worst_case",
        float biasMultiplier_=3.0,
        float wideningParam_=0.3,
        std::string wideningStrategy_="random",
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy=NULL,
        std::string rootAction=""
    );

    void runTrials(
        int batchSize,
        int iterations
    );

    using CvarMCTS::executeEpisode;
    CvarGameHist executeEpisode(
        std::shared_ptr<MDP> pTrueMDP
    );

protected:
    std::shared_ptr<MDP> pTemplateMDP;
    std::shared_ptr<CvarDecisionNode> rootNode;
    State initState;
    float alpha;
    int horizon;
    std::string optim;
    std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy;
    std::string rootAction;
};

#endif
