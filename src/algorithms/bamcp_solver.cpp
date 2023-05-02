/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "mcts_decision_node.h"
#include "hist.h"
#include "bamcp_solver.h"


/* Execute MCTS forward search from a state and return the resulting
root node. */
std::shared_ptr<MCTSDecisionNode> BAMCPSolver::runMCTS(MultiModelMDP& mmdp, State currentState, int horizon, int numTrials, std::shared_ptr<MCTSDecisionNode> pNode){

    // if no decision node is passed as argument create a node for the root
    std::shared_ptr<MDP> pMDP = mmdp.sampleModel();
    std::shared_ptr<MCTSDecisionNode> rootNode;
    if(pNode != NULL){
        rootNode = pNode;
    }else{
        rootNode = std::make_shared<MCTSDecisionNode>(pMDP, currentState);
    }

    MCTSDecisionNode* pCurrentDecisionNode;
    std::vector<MCTSDecisionNode*> decisionNodesToBackup;
    std::vector<MCTSChanceNode*> chanceNodesToBackup;
    std::vector<float> rewards;
    int depth;
    std::shared_ptr<MDP> pModelThisTrial;

    int trial = 0;
    while(trial < numTrials){

        // reset for a new trial
        pCurrentDecisionNode = rootNode.get();
        decisionNodesToBackup.clear();
        chanceNodesToBackup.clear();
        rewards.clear();
        depth = 0;

        // randomly sample a transition model from the belief to be used for
        // this trial
        pModelThisTrial = mmdp.sampleModel();

        // select within tree
        while((depth < horizon) && (pCurrentDecisionNode->isFullyExpanded())){
            std::string action = pCurrentDecisionNode->selectActionUCB(biasFactor);
            pCurrentDecisionNode = select(action, pCurrentDecisionNode, decisionNodesToBackup,
                                        chanceNodesToBackup, rewards, pModelThisTrial);
            depth++;
        }

        // expand by visiting a state action pair which has not yet been visited
        // and performing a random rollout
        if(depth < horizon){
            expand(pCurrentDecisionNode, depth, horizon, decisionNodesToBackup,
                    chanceNodesToBackup, rewards, pModelThisTrial);
        }
        updateNodes(decisionNodesToBackup, chanceNodesToBackup, rewards);
        trial++;
    }
    return rootNode;
}


/* Selects actions within the tree according to the UCB selection criterion.
The function appends the nodes visited and rewards received to the corresponding
vectors

*/
MCTSDecisionNode* BAMCPSolver::select(
        std::string action,
        MCTSDecisionNode* pCurrentDecisionNode,
        std::vector<MCTSDecisionNode*>&  decisionNodes,
        std::vector<MCTSChanceNode*>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP)
{
    MCTSChanceNode* pCurrentChanceNode;
    State nextState;

    // choose action and get next chance node
    pCurrentChanceNode = pCurrentDecisionNode->getChildren()[action].get();

    // record the chance and decision nodes which have been visited
    decisionNodes.push_back(pCurrentDecisionNode);
    chanceNodes.push_back(pCurrentChanceNode);
    rewards.push_back(pCurrentChanceNode->getLocalReward());

    // get state transition and next decision node using the model
    // sampled for this trial
    nextState = pCurrentChanceNode->selectRandomOutcome(pMDP);
    pCurrentDecisionNode = pCurrentChanceNode->getChildren()[nextState].get();
    return pCurrentDecisionNode;
}

/* Expands an unvisited state-action pair and performs a random rollout
from there onwards. The function appends the nodes visited with the expanded
node and also appends the return from the random rollout.

Args:
    pDecisionNode: the decision node at which to expand an action which has not
        been taken.
    depth: the depth of the trial so far.
    decisionNodes: the decision nodes visited in the trial so far.
    chanceNodes: the chance nodes visited in the trial so far.
    rewards: the rewards received in the trial so far.
*/
void BAMCPSolver::expand(
        MCTSDecisionNode* pDecisionNode,
        int depth,
        int horizon,
        std::vector<MCTSDecisionNode*>&  decisionNodes,
        std::vector<MCTSChanceNode*>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP)
{
    MCTSChanceNode* pChanceNode;
    std::string action;

    // choose action to expand
    action = pDecisionNode->expand();
    pChanceNode = pDecisionNode->getChildren()[action].get();

    // simulate a rollout using random actions
    State s = pChanceNode->getState();
    float rolloutReturn = getRolloutReturn(s, action, depth, horizon, pMDP);

    rewards.push_back(rolloutReturn);
    decisionNodes.push_back(pDecisionNode);
    chanceNodes.push_back(pChanceNode);
}

/* Get the return from a random rollout starting with a given state action
pair.

Attributes:
    s: the state
    action: the action corresponding to initial state-action pair.
    depth: the current depth into the episode.
    horizon: the horizon of the episode.
    pMDP: a pointer to an MDP with correct state and action space.

Returns:
    the sum of rewards from a random rollout from this state action pair for
    the remaining steps to the horizon.
*/
float BAMCPSolver::getRolloutReturn(
    State s,
    std::string action,
    int depth,
    int horizon,
    std::shared_ptr<MDP> pMDP
){
    float rolloutReturn = 0.0;
    State nextState;
    while(depth < horizon){
        rolloutReturn += pMDP->getReward(s, action);

        // sample the successor using the transition model for this trial
        nextState = pMDP->sampleSuccessor(s, action);
        action = pMDP->sampleAction(nextState);

        s = nextState;
        depth++;
    }
    return rolloutReturn;
}

/* Updates the average values of all of the nodes visited in a trial.

Args:
    chanceNodesToBackup: vector of chance nodes visited in order.
    decisionNodesToBackup: vector of decision nodes visited in order.
    rewards: vector of rewards visited in order attained.
*/
void BAMCPSolver::updateNodes(
        std::vector<MCTSDecisionNode*>&  decisionNodesToBackup,
        std::vector<MCTSChanceNode*>& chanceNodesToBackup,
        std::vector<float>& rewards)
{
    MCTSDecisionNode* pCurrentDecisionNode;
    MCTSChanceNode* pCurrentChanceNode;

    // backup values through the tree
    float trialReturn = 0.0;
    while(rewards.size() != 0){
        float lastReward;

        // move backwards through list to backup
        lastReward = rewards.back();
        pCurrentChanceNode = chanceNodesToBackup.back();
        pCurrentDecisionNode = decisionNodesToBackup.back();

        // update values along the path
        trialReturn += lastReward;
        pCurrentChanceNode->updateNode(trialReturn);
        pCurrentDecisionNode->updateNode(trialReturn);

        // remove values from vectors to backup
        rewards.pop_back();
        chanceNodesToBackup.pop_back();
        decisionNodesToBackup.pop_back();
    }
}

/* Execute an episode in a Bayes Adaptive MDP using an MCTS solver.

Args:
    mmdp: the multimodel MDP containing the prior over possible MDPs.
    pTrueModel: a shared pointer to the true underlying model used to generate
        transition probabilities.
    initState: the state that the agent starts the episode in.
    horizon: the number of steps before the episode terminates.

Returns:
    history: the history of state-action pairs visited in the episode.
*/
Hist BAMCPSolver::executeEpisode(
        MultiModelMDP& mmdp,
        std::shared_ptr<MDP> pTrueModel,
        State initState,
        int horizon,
        int burnInTrials,
        int trials)
{
    State currentState = initState;
    State nextState;
    std::string action;
    Hist history;
    float reward;
    bool burnIn = true;
    std::shared_ptr<MCTSDecisionNode> pStateNode;
    std::shared_ptr<MCTSChanceNode> pActionNode;
    resetEpisodeReward();

    while(horizon > 0){

        // if this is the first step in the horizon burn in root node otherwise
        // pass existing root node
        if(burnIn){
            pStateNode = runMCTS(mmdp, currentState, horizon, burnInTrials);
            burnIn = false;
        }else{
            pStateNode = runMCTS(mmdp, currentState, horizon, trials, pStateNode);
        }

        // take best action as evaluated by search
        action = pStateNode->getBestAction();
        reward = pTrueModel->getReward(currentState, action);
        pActionNode = pStateNode->getChildren()[action];

        // update solver state for cost incurred so far and append to history
        history.addTransition(currentState, action, reward);
        updateEpisodeReward(reward);

        // sample next state and update belief
        nextState = pTrueModel->sampleSuccessor(currentState, action);
        mmdp.updateBelief(currentState, action, nextState);
        pStateNode = pActionNode->getChildren()[nextState];
        currentState = nextState;
        horizon--;
    }
    return history;
}

/* Estimate the value of the policy corresponding to acting greedily within
the tree. Averages the return over a number of rollouts using greedy action
selection

Arguments:
    mmdp: the mmdp for which we are evaluating the policy
    horizon: the horizon over which to obtain expected value
    numTrials: the number of trials over which to evaluate
    pNode: a shared pointer to the root node of the tree to evaluate
*/
float BAMCPSolver::estimateValue(
    MultiModelMDP& mmdp,
    int horizon,
    int numTrials,
    std::shared_ptr<MCTSDecisionNode> pRootNode
){
    std::shared_ptr<MDP> pMDP = mmdp.sampleModel();
    MCTSDecisionNode* pCurrentDecisionNode;
    std::vector<MCTSDecisionNode*> decisionNodesToBackup;
    std::vector<MCTSChanceNode*> chanceNodesToBackup;
    std::vector<float> rewards;
    int depth;
    std::shared_ptr<MDP> pModelThisTrial;

    int trial = 0;
    float returnSum = 0.0;
    while(trial < numTrials){

        // reset for a new trial
        pCurrentDecisionNode = pRootNode.get();
        decisionNodesToBackup.clear();
        chanceNodesToBackup.clear();
        rewards.clear();
        depth = 0;

        // randomly sample a transition model from the belief to be used for
        // this trial
        pModelThisTrial = mmdp.sampleModel();

        // select within tree
        while((depth < horizon) && (pCurrentDecisionNode->isFullyExpanded())){
            std::string action = pCurrentDecisionNode->getBestAction();
            pCurrentDecisionNode = select(action, pCurrentDecisionNode, decisionNodesToBackup,
                                        chanceNodesToBackup, rewards, pModelThisTrial);
            depth++;
        }

        if(depth < horizon){
            State s = pCurrentDecisionNode->getState();
            std::string action = pMDP->sampleAction(s);
            float rolloutReturn = getRolloutReturn(s, action, depth, horizon, pMDP);
            rewards.push_back(rolloutReturn);
        }

        returnSum += getReturn(rewards);
        trial++;
    }
    return returnSum/numTrials;
}

/* Return the total value corresponding to the vector of rewards collected.
*/
float BAMCPSolver::getReturn(
    std::vector<float>& rewards
){
    float totalReturn = 0.0;
    for(auto rew : rewards){
        totalReturn += rew;
    }
    return totalReturn;
}

void BAMCPSolver::resetEpisodeReward() {
    rewardThisEpisode = 0.0;
}

void BAMCPSolver::updateEpisodeReward(float reward) {
    rewardThisEpisode += reward;
}
