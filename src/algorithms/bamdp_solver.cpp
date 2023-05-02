/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "hist.h"
#include "bamdp_solver.h"


/* Execute an episode in a Bayes Adaptive MDP.

Args:
    mmdp: the multimodel MDP containing the prior over possible MDPs.
    pTrueModel: a shared pointer to the true underlying model used to generate
        transition probabilities.
    initState: the state that the agent starts the episode in.
    horizon: the number of steps before the episode terminates.

Returns:
    history: the history of state-action pairs visited in the episode.
*/
Hist BAMDPSolver::executeEpisode(
        MultiModelMDP& mmdp,
        std::shared_ptr<MDP> pTrueModel,
        State initState,
        int horizon)
{
    State currentState = initState;
    State nextState;
    std::string action;
    Hist history;
    float reward;
    resetEpisodeReward();

    while(horizon > 0){
        action = this->getNextAction(mmdp, currentState, horizon);
        reward = pTrueModel->getReward(currentState, action);

        // update solver state for cost incurred so far and append to history
        history.addTransition(currentState, action, reward);
        updateEpisodeReward(reward);

        // sample next state and update belief
        nextState = pTrueModel->sampleSuccessor(currentState, action);
        mmdp.updateBelief(currentState, action, nextState);
        currentState = nextState;
        horizon--;
    }
    return history;
}

void BAMDPSolver::resetEpisodeReward() {
    rewardThisEpisode = 0.0;
}

void BAMDPSolver::updateEpisodeReward(float reward) {
    rewardThisEpisode += reward;
}
