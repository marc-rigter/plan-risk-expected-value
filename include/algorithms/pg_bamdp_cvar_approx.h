#ifndef cvar_pg_approx
#define cvar_pg_approx
#include <string>
#include "state.h"
#include "pg_decision_node.h"
#include "belief.h"
#include "pg_bamdp_cvar.h"
#include "finite_mdp_belief.h"


/* Class to implement the policy gradient approach to optimise
*/
class BamdpCvarPGApprox : public BamdpCvarPG
{
protected:
    std::vector<std::vector<float>> W;
    std::shared_ptr<FiniteMDPBelief> rootParticleBelief;
    std::unordered_map<std::shared_ptr<MDP>, int> mdpMapping;


public:
    BamdpCvarPGApprox(
            float learningRate_,
            std::shared_ptr<Belief> pBelief,
            State initState_,
            float alpha_,
            int horizon_,
            bool initCvarPolicy
    );

    void initMatrixWithCvarPolicy(
        std::shared_ptr<Belief> pBelief
    );

    void runTrials(
        int batchSize,
        int iterations
    );

    std::string selectActionSoftmax(
        State currentState,
        std::shared_ptr<FiniteMDPBelief> currentBelief,
        std::shared_ptr<MDP> pTemplateMDP,
        std::unordered_map<State, int, StateHash>& stateMapping,
        std::unordered_map<std::string, int>& actionMapping
    );

    std::shared_ptr<FiniteMDPBelief> getNewBelief(
            std::shared_ptr<FiniteMDPBelief> currentBelief,
            State s,
            std::string action,
            State nextState
    );

    int getStateActionIndex(
            State s,
            std::string act,
            std::unordered_map<State, int, StateHash>& stateMapping,
            std::unordered_map<std::string, int>& actionMapping
    );

    void gradientUpdate(
            std::vector<std::vector<State>>& trajectoryStates,
            std::vector<std::vector<std::string>>& trajectoryActions,
            std::vector<std::vector<std::shared_ptr<FiniteMDPBelief>>>& trajectoryBeliefs,
            std::vector<float>& returns,
            std::shared_ptr<MDP> pTemplateMDP,
            std::unordered_map<State, int, StateHash>& stateMapping,
            std::unordered_map<std::string, int>& actionMapping
    );

    Hist executeEpisode(std::shared_ptr<MDP> pTrueModel = NULL);
};

#endif
