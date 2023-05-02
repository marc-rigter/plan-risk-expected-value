#ifndef bamcp_solver
#define bamcp_solver
#include <string>
#include "state.h"
#include "multimodel_mdp.h"
#include "bamdp_solver.h"
#include "mcts_decision_node.h"
#include "mcts_chance_node.h"


/* Class to implement the Bayes adaptive Monte Carlo search approach to solving
Bayes adaptive MDPs.

Attributes:
    maxTrials: the maximum number of trials to be used to find the next action.
*/
class BAMCPSolver {
protected:
    float biasFactor;
    float rewardThisEpisode = 0.0;

public:
    //BAMCPSolver(){};
    BAMCPSolver(float biasFactor_=3.0) : biasFactor(biasFactor_) {};
    void resetEpisodeReward();
    void updateEpisodeReward(float reward);

    Hist executeEpisode(
        MultiModelMDP& mmdp,
        std::shared_ptr<MDP> pTrueModel,
        State currentState,
        int horizon,
        int burnInTrials,
        int trials
    );

    float getRolloutReturn(
        State s,
        std::string action,
        int depth,
        int horizon,
        std::shared_ptr<MDP> pMDP
    );

    std::shared_ptr<MCTSDecisionNode> runMCTS(
        MultiModelMDP& mmdp,
        State currentState,
        int horizon,
        int numTrials,
        std::shared_ptr<MCTSDecisionNode> pNode=nullptr
    );

    float estimateValue(
        MultiModelMDP& mmdp,
        int horizon,
        int numTrials,
        std::shared_ptr<MCTSDecisionNode> pNode
    );

    virtual void updateNodes(
        std::vector<MCTSDecisionNode*>& dNodes,
        std::vector<MCTSChanceNode*>& cNodes,
        std::vector<float>& rewards
    );

    virtual float getReturn(
        std::vector<float>& rewards
    );

    void expand(
        MCTSDecisionNode* pDecisionNode,
        int depth,
        int horizon,
        std::vector<MCTSDecisionNode*>&  decisionNodes,
        std::vector<MCTSChanceNode*>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP
    );

    MCTSDecisionNode* select(
        std::string action,
        MCTSDecisionNode* pCurrentDecisionNode,
        std::vector<MCTSDecisionNode*>&  decisionNodes,
        std::vector<MCTSChanceNode*>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP
    );
};

#endif
