#ifndef mcts_cvar_sg
#define mcts_cvar_sg
#include <string>
#include "state.h"
#include "mcts.h"
#include "cvar_chance_node.h"
#include "cvar_decision_node.h"
#include "cvar_game_hist.h"
#include "bamdp_rollout_policy.h"

/* Class to implement monte carlo tree search to solve game optimising CVaR
in MDPs
*/
class CvarMCTS : public MCTS {
public:
    CvarMCTS(float biasMultiplier_=3.0, float wideningParam_=0.3, std::string wideningStrategy_="random");

    void cvarMCTS(
        std::shared_ptr<MDP> pMDP,
        std::shared_ptr<CvarDecisionNode> pNode,
        int horizon,
        int numTrials,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy=NULL,
        std::string rootAction=""
    );

    float trial(
        std::shared_ptr<MDP> pTemplateMDP,
        std::shared_ptr<CvarDecisionNode> rootNode,
        int horizon,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy=NULL,
        std::string rootAction=""
    );

    void expand(
        std::shared_ptr<CvarDecisionNode> pAgentNode,
        int depth,
        int horizon,
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodes,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodes,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy
    );

    float getRolloutReturn(
        std::shared_ptr<CvarChanceNode> chanceNode,
        int depth,
        int horizon,
        std::shared_ptr<MDP> pMDP,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy
    );

    std::shared_ptr<CvarDecisionNode> select(
        std::shared_ptr<CvarDecisionNode> pCurrentAgentNode,
        int depth,
        int horizon,
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodes,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodes,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP,
        std::string overrideAction=""
    );

    float updateNodes(
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodesToBackup,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodesToBackup,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodesToBackup,
        std::vector<float>& rewards
    );

    CvarGameHist executeEpisode(
        std::shared_ptr<MDP> pMDP,
        std::shared_ptr<CvarDecisionNode> rootNode,
        int horizon,
        int burnInTrials,
        int agentTrials,
        int advTrials,
        std::string optim = "worst_case",
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy=NULL
    );

    void finishEpisodeRandom(
        std::shared_ptr<CvarDecisionNode> pAgentNode,
        int horizon,
        std::shared_ptr<MDP> pMDP,
        CvarGameHist& history
    );

    int expandAction(
        std::shared_ptr<CvarAdvDecisionNode> pAdversaryNode,
        int stepsRemaining
    );

protected:
    float wideningParam;
    std::string wideningStrategy;
};

#endif
