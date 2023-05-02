/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <random>
#include <algorithm>
#include "pg_bamdp_cvar.h"
#include "pg_chance_node.h"
#include "pg_decision_node.h"
#include "hist.h"


BamdpCvarPG::BamdpCvarPG(
        float learningRate_,
        std::shared_ptr<Belief> pBelief_,
        State initState_,
        float alpha_,
        int horizon_
) : learningRate(learningRate_), rootBelief(pBelief_), initState(initState_), alpha(alpha_), horizon(horizon_)
{
    // set the root node to point to node at initials state
    std::shared_ptr<MDP> pTemplateMDP = pBelief_->sampleModel();
    rootNode = std::make_shared<PGDecisionNode>(pTemplateMDP, initState);

    // computing a new mdp for root sampling each trial is expensive. so
    // precompute a number of models to store in memory here
    int numModels = 200;
    for(int i = 0; i < numModels; i++){
        rootSamples.push_back(pBelief_->sampleModel());
    }
};

Hist BamdpCvarPG::executeEpisode(std::shared_ptr<MDP> pTrueModel){
    Hist history;
    std::shared_ptr<PGDecisionNode> pDecisionNode;
    std::shared_ptr<PGChanceNode> pChanceNode;

    int depth = 0;
    pDecisionNode = rootNode;
    State currentState = rootNode->getState();
    while(depth < horizon){

        // select action and record reward received
        std::string action = pDecisionNode->selectActionSoftmax();
        pChanceNode = pDecisionNode->getChildren()[action];
        float reward = pTrueModel->getReward(currentState, action);
        history.addTransition(currentState, action, reward);

        // sample a successor using the model for this trial
        State nextState = pChanceNode->sampleSuccessor(pTrueModel);

        // update for next step
        pDecisionNode = pChanceNode->getChildren()[nextState];
        currentState = nextState;
        depth++;
    }

    return history;
}

/* Execute policy gradient updates */
void BamdpCvarPG::runTrials(
    int batchSize,
    int iterations
){

    // get a template MDP to use for enabled actions and rewards
    std::shared_ptr<MDP> pTemplateMDP = rootBelief->sampleModel();

    // random int generator to sample models from memory
    int numModels = rootSamples.size();
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, numModels-1);

    std::shared_ptr<PGDecisionNode> pDecisionNode;
    std::shared_ptr<PGChanceNode> pChanceNode;

    int it = 0;
    while(it < iterations){

        // reset variables for this batch
        std::vector<std::vector<std::shared_ptr<PGDecisionNode>>> trajectories;
        std::vector<std::vector<std::string>> trajectoryActions;
        std::vector<float> returns;
        int trial = 0;

        while(trial < batchSize){

            // sample the model for this trial to use root sampling
            int modelIndex = uni(rng);
            std::shared_ptr<MDP> pModelThisTrial = rootSamples.at(modelIndex);

            // reset variables for this trial
            float rewardThisEpisode = 0.0;
            int depth = 0;
            pDecisionNode = rootNode;
            State currentState = initState;
            std::vector<std::shared_ptr<PGDecisionNode>> traj;
            std::vector<std::string> trajActions;

            while(depth < horizon){

                // select action and record reward received
                std::string action = pDecisionNode->selectActionSoftmax();
                pChanceNode = pDecisionNode->getChildren()[action];
                float reward = pTemplateMDP->getReward(currentState, action);
                traj.push_back(pDecisionNode);
                trajActions.push_back(action);
                rewardThisEpisode += reward;

                // sample a successor using the model for this trial
                State nextState = pChanceNode->sampleSuccessor(pModelThisTrial);

                // update for next step
                pDecisionNode = pChanceNode->getChildren()[nextState];
                currentState = nextState;
                depth++;
            }
            // store the result of this trial
            trajectories.push_back(traj);
            trajectoryActions.push_back(trajActions);
            returns.push_back(rewardThisEpisode);

            trial++;
        }

        // compute the empirical estimate of the VaR
        std::vector<float> returnsSorted= returns;
        std::sort(returnsSorted.begin(), returnsSorted.end());
        int varIndex = (int)(std::round(batchSize * alpha)) - 1;
        float varEstimate = returnsSorted[varIndex];
        float cvarSum = 0.0;
        for(int i = 0; i <= varIndex; i++){
            cvarSum += returnsSorted[i];
        }
        std::cout << "Iteration: " << it << ", Batch empirical cvar: " << cvarSum/(varIndex + 1) << std::endl;
        gradientUpdate(
            trajectories,
            trajectoryActions,
            returns,
            varEstimate
        );
        it++;
    }
}


void BamdpCvarPG::gradientUpdate(
        std::vector<std::vector<std::shared_ptr<PGDecisionNode>>> trajectories,
        std::vector<std::vector<std::string>> trajectoryActions,
        std::vector<float> returns,
        float varEstimate
){
    // compute the gradient
    std::unordered_map<std::shared_ptr<PGDecisionNode>, std::unordered_map<std::string, float>> gradients;
    for(unsigned int i = 0; i < trajectories.size(); i++){
        if(returns[i] >  varEstimate){
            continue;
        }
        std::vector<std::shared_ptr<PGDecisionNode>> traj = trajectories[i];
        std::vector<std::string> trajActions = trajectoryActions[i];
        float returnThisTraj = returns[i];

        for(unsigned int j = 0; j < traj.size(); j++){
            std::shared_ptr<PGDecisionNode> pDecisionNode = traj[j];
            std::string actionTaken = trajActions[j];
            std::unordered_map<std::string, float> actionWeights;
            actionWeights = pDecisionNode->getActionWeights();

            // if this decision node is not in the mapping add it
            if (gradients.find(pDecisionNode) == gradients.end()){
                std::unordered_map<std::string, float> gradient;
                for(auto kv : pDecisionNode->getActionWeights()){
                    gradient[kv.first] = 0.0;
                }
                gradients[pDecisionNode] = gradient;
            }

            // get the sum for the softmax denominator
            float expSum = 0.0;
            for(auto pair : actionWeights){
                expSum += std::exp(pair.second);
            }

            // gradient computation for this node
            for(auto pair : actionWeights){
                if(pair.first == actionTaken){
                    gradients[pDecisionNode][pair.first] += (1 - std::exp(pair.second)/expSum)*(returnThisTraj - varEstimate)/alpha;
                }else{
                    gradients[pDecisionNode][pair.first] += -std::exp(pair.second)/expSum*(returnThisTraj - varEstimate)/alpha;
                }
            }
        }
    }

    // now that the gradient has been computed update the values at each decision node
    for(auto pair : gradients){
        std::shared_ptr<PGDecisionNode> pNode = pair.first;
        std::unordered_map<std::string, float> gradient = pair.second;
        std::unordered_map<std::string, float> actionWeights = pNode->getActionWeights();
        for(auto kv : actionWeights){
            pNode->setActionWeight(kv.first, kv.second + gradient[kv.first] * learningRate);
        }
    }
}
