/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include "mdp.h"
#include "state.h"
#include "cvar_game_hist.h"
#include "mcts_cvar_sg.h"
#include "cvar_chance_node.h"
#include "cvar_adv_decision_node.h"
#include "bamdp_cvar_decision_node.h"
#include "bamdp_cvar_adv_decision_node.h"
#include "bamdp_rollout_policy.h"
#include "utils.h"


CvarMCTS::CvarMCTS(float biasMultiplier_, float wideningParam_, std::string wideningStrategy_)
    : MCTS(biasMultiplier_), wideningParam(wideningParam_), wideningStrategy(wideningStrategy_)
{

}

/* Execute MCTS forward search from a state and return the resulting
root node. */
void CvarMCTS::cvarMCTS(
        std::shared_ptr<MDP> pTemplateMDP,
        std::shared_ptr<CvarDecisionNode> rootNode,
        int horizon,
        int numTrials,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy,
        std::string rootAction)
{
    int trial = 0;

    while(trial < numTrials){
        this->trial(
            pTemplateMDP,
            rootNode,
            horizon,
            rolloutPolicy,
            rootAction
        );
        trial++;
    }
}

/* Run a single trial of MCTS */
float CvarMCTS::trial(
    std::shared_ptr<MDP> pTemplateMDP,
    std::shared_ptr<CvarDecisionNode> rootNode,
    int horizon,
    std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy,
    std::string rootAction
){
    std::shared_ptr<CvarDecisionNode> pCurrentAgentNode;
    std::vector<std::shared_ptr<CvarDecisionNode>> agentNodesToBackup;
    std::vector<std::shared_ptr<CvarAdvDecisionNode>> adversaryNodesToBackup;
    std::vector<std::shared_ptr<CvarChanceNode>> chanceNodesToBackup;
    std::vector<float> rewards;
    int depth;

    pCurrentAgentNode = rootNode;
    depth = 0;

    // select nodes within the tree until a node which is not fully
    // expanded is reached
    while((depth < horizon) && (pCurrentAgentNode->isFullyExpanded())){

        // if at the root node use the action supplied for the root node
        if(pCurrentAgentNode == rootNode){
            pCurrentAgentNode = select(pCurrentAgentNode,
                                    depth,
                                    horizon,
                                    agentNodesToBackup,
                                    adversaryNodesToBackup,
                                    chanceNodesToBackup,
                                    rewards,
                                    pTemplateMDP,
                                    rootAction);

        // otherwise use ucb for the root node
        }else{
            pCurrentAgentNode = select(pCurrentAgentNode,
                                    depth,
                                    horizon,
                                    agentNodesToBackup,
                                    adversaryNodesToBackup,
                                    chanceNodesToBackup,
                                    rewards,
                                    pTemplateMDP);
        }

        depth++;
    }

    // expand by visiting a state-action-action pair which has not yet been
    // visited and performing a random rollout
    if(depth < horizon){
        State s = pCurrentAgentNode->getState();
        expand(
            pCurrentAgentNode,
            depth,
            horizon,
            agentNodesToBackup,
            adversaryNodesToBackup,
            chanceNodesToBackup,
            rewards,
            pTemplateMDP,
            rolloutPolicy
        );
    }

    // update counts and average values at nodes
    float trialReturn = updateNodes(
                            agentNodesToBackup,
                            adversaryNodesToBackup,
                            chanceNodesToBackup,
                            rewards);
    return trialReturn;
}

/* Execute the stochastic cvar game using the MCTS solver. Note that during
execution we just sample transitions according to the true underlying MDP
provided (as the adversary actions are really just fictitious).

Args:
    pTrueMDP: a pointer to the true MDP used to generate transitiosn.
    rootNode: the root node to being the episode from
    horizon: the horizon for the episode
    burnInTrials: the number of MCTS trials at the root node
    trials: the number of MCTS trials at subsequent nodes.
    optim: string defining how to handle cases where the path probability gets
        set to zero. options: "worst_case" - switch to optimising for the worst
        case, "expected_value" - switch to optimising for the expected value,
        "random" - finish the episode randomly.
    rolloutPolicy: pointer to rollout policy to use for rollout trials.
*/
CvarGameHist CvarMCTS::executeEpisode(
        std::shared_ptr<MDP> pTrueMDP,
        std::shared_ptr<CvarDecisionNode> rootNode,
        int horizon,
        int burnInTrials,
        int agentTrials,
        int advTrials,
        std::string optim,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy
)
{
    State currentState = rootNode->getState();
    State nextState;
    std::string action;
    CvarGameHist history;
    float reward;

    std::shared_ptr<CvarDecisionNode> pCurrentAgentNode = rootNode;
    std::shared_ptr<CvarAdvDecisionNode> pCurrentAdvNode;
    std::shared_ptr<CvarChanceNode> pCurrentChanceNode;
    resetEpisodeReward();

    bool burnIn = true;
    while(horizon > 0){

        // if this is the first step in the horizon burn in root node otherwise
        // pass existing root node
        if(burnIn){
            cvarMCTS(pTrueMDP, pCurrentAgentNode, horizon, burnInTrials, rolloutPolicy);
            burnIn = false;
        }else{
            cvarMCTS(pTrueMDP, pCurrentAgentNode, horizon, agentTrials, rolloutPolicy);
        }

        // ensure that the node is expanded and there is an action available
        if(!pCurrentAgentNode->isExpanded()){
            cvarMCTS(pTrueMDP, pCurrentAgentNode, horizon, 1, rolloutPolicy);
        }

        // take best agent action as evaluated by search
        action = pCurrentAgentNode->getBestAction();
        reward = pTrueMDP->getReward(currentState, action);

        // perform additional rollouts starting from the corresponding adv node
        pCurrentAdvNode = pCurrentAgentNode->getAdvChildren()[action];
        cvarMCTS(pTrueMDP, pCurrentAgentNode, horizon, advTrials, rolloutPolicy, action);

        // choose best adversary action
        int advActionInd = pCurrentAdvNode->getBestAdvAction();
        std::unordered_map<State, float, StateHash> advPerturbation = pCurrentAdvNode->getPerturbationMapping(advActionInd);

        // select associated chance node
        pCurrentChanceNode = pCurrentAdvNode->getChanceChildren()[advActionInd];

        // update solver state for cost incurred so far and append to history
        history.addTransition(pCurrentAgentNode, action, advActionInd, reward);
        updateEpisodeReward(reward);

        // sample next state and update belief
        nextState = pTrueMDP->sampleSuccessor(currentState, action);
        pCurrentChanceNode->addChild(nextState);
        pCurrentAgentNode = pCurrentChanceNode->getCvarChildren()[nextState];

        // if set to finish episodes randomly when alpha is zero.
        if(cmpf(pCurrentAgentNode->getAlpha(), 0.0, 1e-6)){
            if(optim == "random"){
                std::cout << pCurrentAgentNode << std::endl;
                finishEpisodeRandom(pCurrentAgentNode, horizon, pTrueMDP, history);
                break;

            // if changing to optimise expected value set alpha
            }else if(optim == "expected_value"){
                pCurrentAgentNode->setAlpha(1.0);

            // for worst case will automatically be optimised if alpha is zero
            }else if(optim != "worst_case" ){
                throw "Error in optimisation option";
            }
        }

        // if the rollout policy is not null update it according to currennt belief
        if(rolloutPolicy != NULL){
            rolloutPolicy->updateRolloutPolicy(dynamic_cast<BamdpCvarDecisionNode*>(pCurrentAgentNode.get())->getBelief());
        }

        currentState = nextState;
        horizon--;
    }
    return history;
}


/* Updates the average values of all of the nodes visited in a trial.

Args:
    chanceNodesToBackup: vector of chance nodes visited in order.
    agentNodesToBackup: vector of cvar decision nodes visited in order.
    advesraryNodesToBackup: vector of adv decision nodes visited in order.
    rewards: vector of rewards visited in order attained.

Returns:
    the total return from the trial
*/
float CvarMCTS::updateNodes(
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodesToBackup,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodesToBackup,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodesToBackup,
        std::vector<float>& rewards)
{
    std::shared_ptr<CvarDecisionNode> pCurrentAgentNode;
    std::shared_ptr<CvarAdvDecisionNode> pCurrentAdvNode;
    std::shared_ptr<CvarChanceNode> pCurrentChanceNode;

    // backup values through the tree
    float trialReturn = 0.0;
    while(rewards.size() != 0){
        float lastReward;

        // move backwards through list to backup
        lastReward = rewards.back();
        pCurrentChanceNode = chanceNodesToBackup.back();
        pCurrentAgentNode = agentNodesToBackup.back();
        pCurrentAdvNode = adversaryNodesToBackup.back();

        // update values along the path
        trialReturn += lastReward;
        pCurrentChanceNode->updateNode(trialReturn);
        pCurrentAgentNode->updateNode(trialReturn);
        pCurrentAdvNode->updateNode(trialReturn);

        // remove values from vectors to backup
        rewards.pop_back();
        chanceNodesToBackup.pop_back();
        agentNodesToBackup.pop_back();
        adversaryNodesToBackup.pop_back();
    }

    return trialReturn;
}

/* Selects actions within the tree according to the UCB selection criterion.
The function appends the nodes visited and rewards received to the corresponding
vectors.
*/
std::shared_ptr<CvarDecisionNode> CvarMCTS::select(
        std::shared_ptr<CvarDecisionNode> pCurrentAgentNode,
        int depth,
        int horizon,
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodes,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodes,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP,
        std::string overrideAction)
{
    std::shared_ptr<CvarAdvDecisionNode> pCurrentAdvNode;

    // ucb for agent action
    std::string action;
    if(overrideAction == ""){
        action = pCurrentAgentNode->selectActionUCB(biasFactor);
    }else{
        action = overrideAction;
    }
    pCurrentAdvNode = pCurrentAgentNode->getAdvChildren()[action];

    // compute the bounded number of actions at this adversary node according
    // to progressive widening.
    unsigned int maxActions = ceil(pow(pCurrentAdvNode->getObservations(), wideningParam));
    int advActionInd;
    if(pCurrentAdvNode->getChanceChildren().size() < maxActions){

        // expand new action
        int stepsRemaining = horizon - depth;
        advActionInd = expandAction(pCurrentAdvNode, stepsRemaining);
    }else{
        advActionInd = pCurrentAdvNode->selectPerturbationActionUCB(biasFactor);
    }

    std::unordered_map<State, float, StateHash> advPerturbation;
    advPerturbation = pCurrentAdvNode->getPerturbationMapping(advActionInd);

    // get the chance node associated with this state and two actions
    std::shared_ptr<CvarChanceNode> pCurrentChanceNode;
    pCurrentChanceNode = pCurrentAdvNode->getChanceChildren()[advActionInd];

    // record the chance and decision nodes which have been visited
    agentNodes.push_back(pCurrentAgentNode);
    adversaryNodes.push_back(pCurrentAdvNode);
    chanceNodes.push_back(pCurrentChanceNode);
    rewards.push_back(pCurrentChanceNode->getLocalReward());

    // get state transition and next decision node using the model
    // and the perturbation chosen
    State currentState, nextState;
    currentState = pCurrentAgentNode->getState();
    nextState = pCurrentChanceNode->sampleSuccessorAddChild();
    pCurrentAgentNode = pCurrentChanceNode->getCvarChildren()[nextState];
    return pCurrentAgentNode;
}

/* Call the correct function to expand an adversary action according to the
strategy for choosing the adversary action */
int CvarMCTS::expandAction(
    std::shared_ptr<CvarAdvDecisionNode> pAdversaryNode,
    int stepsRemaining
){
    int advActionInd;

    // if strategy uses heuristic first action expanded should use rollout policy
    if(wideningStrategy=="bayesOpt"){
        advActionInd = pAdversaryNode->expandPerturbationBayesOpt();
    }else if(wideningStrategy=="random"){
        advActionInd = pAdversaryNode->expandRandomPerturbation();
    }else{
        std::cout << "widening strat: " << wideningStrategy << std::endl;
        std::cerr << "Invalid widening strategy. Options are heuristicBayesOpt, bayesOpt or random." << std::endl;
        std::exit(-1);
    }
    return advActionInd;
}

/* Expands an unvisited state-action-adversary action tuple and then performs a
random rollout from there onwards. The function appends the nodes visited with
the expanded node and also appends the return from the random rollout.

Args:
    pDecisionNode: the decision node at which to expand an action which has not
        been taken.
    depth: the depth of the trial so far.
    decisionNodes: the decision nodes visited in the trial so far.
    chanceNodes: the chance nodes visited in the trial so far.
    rewards: the rewards received in the trial so far.
*/
void CvarMCTS::expand(
        std::shared_ptr<CvarDecisionNode> pAgentNode,
        int depth,
        int horizon,
        std::vector<std::shared_ptr<CvarDecisionNode>>&  agentNodes,
        std::vector<std::shared_ptr<CvarAdvDecisionNode>>&  adversaryNodes,
        std::vector<std::shared_ptr<CvarChanceNode>>& chanceNodes,
        std::vector<float>& rewards,
        std::shared_ptr<MDP> pMDP,
        std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy)
{
    std::shared_ptr<CvarChanceNode> pChanceNode;
    std::shared_ptr<CvarAdvDecisionNode> pAdversaryNode;
    std::string action;
    int advActionInd;
    std::unordered_map<State, float, StateHash> advPerturbation;

    // choose action for the agent to expand
    action = pAgentNode->expand();
    pAdversaryNode = pAgentNode->getAdvChildren()[action];

    // choose action for the adversary to expand
    int stepsRemaining = horizon - depth;
    advActionInd = expandAction(pAdversaryNode, stepsRemaining);
    pChanceNode = pAdversaryNode->getChanceChildren()[advActionInd];
    advPerturbation = pAdversaryNode->getPerturbationMapping(advActionInd);

    // simulate a rollout using random actions
    float rolloutReturn = getRolloutReturn(pChanceNode, depth, horizon, pMDP, rolloutPolicy);

    rewards.push_back(rolloutReturn);
    agentNodes.push_back(pAgentNode);
    adversaryNodes.push_back(pAdversaryNode);
    chanceNodes.push_back(pChanceNode);
}

/* Finish and episode by choosing random actions, and under no perturbations
from the adversary.

Arguments:
    pAgentNode: shared pointer to the agent node from which to begin random
        actions until the end of the episode.
    depth: the current depth into the episode.
    horizon: the fiinal horizon of the episode.
    pTrueMDP: the true underlying MDP which dictates the transition probs
        for the episode.
    history: a reference to the current CVaR game history. The remainder of the
        random rollout will be appended to this history.

*/
void CvarMCTS::finishEpisodeRandom(
    std::shared_ptr<CvarDecisionNode> pAgentNode,
    int horizon,
    std::shared_ptr<MDP> pTrueMDP,
    CvarGameHist& history
){
    std::shared_ptr<CvarAdvDecisionNode> pCurrentAdvNode;
    std::shared_ptr<CvarChanceNode> pCurrentChanceNode;
    State s = pAgentNode->getState();
    while(horizon > 0){

        // choose actions randomly
        std::string action = pTrueMDP->sampleAction(s);
        float reward = pTrueMDP->getReward(s, action);

        // ensure all actions expanded with child node at the agent node
        while(!pAgentNode->isFullyExpanded()){
            pAgentNode->expand();
        }
        pCurrentAdvNode = pAgentNode->getAdvChildren()[action];

        // get adversary action corresponding to no perturbation
        std::unordered_map<State, float, StateHash> noPerturbation = pCurrentAdvNode->getNoPerturbationMapping();
        int advAction = pCurrentAdvNode->addChildChanceNode(noPerturbation);
        pCurrentChanceNode = pCurrentAdvNode->getChanceChildren()[advAction];

        // record transition in history
        history.addTransition(pAgentNode, action, advAction, reward);

        // sample random successor according to true MDP probs
        s = pTrueMDP->sampleSuccessor(s, action);
        pCurrentChanceNode->addChild(s);
        pAgentNode = pCurrentChanceNode->getCvarChildren()[s];
        horizon--;
    }
}

/* Get the return from a random rollout starting with a given state, action, and
adversary action. During the rollout the agent takes random actions and
the adversary applies random perturbations within the budget.

Arguments:
    chanceNode: the chance node from which to begin the rollout.
    depth: the current depth into the episode.
    horizon: the horizon of the episode.
    pMDP: a pointer to an MDP with correct state and action space.

Returns:
    the sum of rewards from a random rollout from this state action pair for
    the remaining steps to the horizon.
*/
float CvarMCTS::getRolloutReturn(
    std::shared_ptr<CvarChanceNode> pChanceNode,
    int depth,
    int horizon,
    std::shared_ptr<MDP> pMDP,
    std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy
){
    State s = pChanceNode->getState();
    std::string action = pChanceNode->getAction();
    std::unordered_map<State, float, StateHash> advMapping = pChanceNode->getPerturbation();

    float rolloutReturn = 0.0;
    State nextState;

    std::shared_ptr<CvarDecisionNode> pDecisionNode;
    std::shared_ptr<CvarAdvDecisionNode> pAdversaryNode;
    while(depth < horizon){

        // collect reward
        rolloutReturn += pMDP->getReward(s, action);

        // sample the successor state according to the perturbation applied by adversary
        nextState = pChanceNode->sampleSuccessor();
        pDecisionNode = pChanceNode->createChild(nextState);

        // if there is no rollout policy sample action and perturbation randomly
        if(rolloutPolicy == NULL){
            action = pMDP->sampleAction(nextState);
            pAdversaryNode = pDecisionNode->createChild(action);
            std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> tup = pAdversaryNode->sampleRandomPerturbations(1);
            advMapping = std::get<1>(tup).front();
        }else{

            // if alpha becomes set to zero rollout optimising expected value
            float currentAlpha = pDecisionNode->getAlpha();
            std::shared_ptr<Belief> pCurrentBelief = dynamic_cast<BamdpCvarDecisionNode*>(pDecisionNode.get())->getBelief();
            action = rolloutPolicy->getRolloutAction(nextState, currentAlpha, horizon-depth, pCurrentBelief);
            pAdversaryNode = pDecisionNode->createChild(action);
            std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> tup = pAdversaryNode->sampleRandomPerturbations(1);
            advMapping = std::get<1>(tup).front();
        }

        pChanceNode = pAdversaryNode->createChild(advMapping);

        // update state
        s = nextState;
        depth++;
    }
    return rolloutReturn;
}
