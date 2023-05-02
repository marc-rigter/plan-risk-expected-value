/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <random>
#include <algorithm>
#include "pg_bamdp_cvar_approx.h"
#include "pg_chance_node.h"
#include "pg_decision_node.h"
#include "finite_mdp_belief.h"
#include "cvar_value_iteration.h"

BamdpCvarPGApprox::BamdpCvarPGApprox(
        float learningRate_,
        std::shared_ptr<Belief> pBelief,
        State initState_,
        float alpha_,
        int horizon_,
        bool initCvarPolicy_
) : BamdpCvarPG(learningRate_, pBelief, initState_, alpha_, horizon_)
{
    // use some of the randomly sampled models as particles
    int numParticles = std::min(25, pBelief->getMaxNumSamples());
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights = pBelief->getParticleWeights(numParticles);
    int n = 0;
    for(auto kv : mdpWeights){
        mdpMapping[kv.first] = n;
        n++;
    }
    rootParticleBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    // initialise the weight matrix
    std::shared_ptr<MDP> pTemplateMDP = pBelief->sampleModel();
    std::unordered_map<std::string, int> mdpActionMap = pTemplateMDP->getActionMapping();
    std::unordered_map<State, int, StateHash> mdpStateMap = pTemplateMDP->getStateMapping();
    std::vector<std::vector<float>> mat(numParticles, std::vector<float>(mdpActionMap.size()*mdpStateMap.size(), 1.0));
    W = mat;

    if(initCvarPolicy_){
        initMatrixWithCvarPolicy(pBelief);
    }
};


Hist BamdpCvarPGApprox::executeEpisode(std::shared_ptr<MDP> pTrueModel){
    std::shared_ptr<MDP> pModelThisTrial;
    if(pTrueModel == NULL){
        int numModels = rootSamples.size();
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, numModels-1);
        int modelIndex = uni(rng);
        pModelThisTrial = rootSamples.at(modelIndex);
    }else{
        pModelThisTrial = pTrueModel;
    }

    Hist history;
    int depth = 0;
    State currentState = initState;
    std::shared_ptr<FiniteMDPBelief> currentBelief = rootParticleBelief;
    std::unordered_map<std::string, int> mdpActionMap = pModelThisTrial->getActionMapping();
    std::unordered_map<State, int, StateHash> mdpStateMap = pModelThisTrial->getStateMapping();

    while(depth < horizon){

        // select action and record reward received
        std::string action = selectActionSoftmax(
                currentState,
                currentBelief,
                pModelThisTrial,
                mdpStateMap,
                mdpActionMap
            );

        float reward = pModelThisTrial->getReward(currentState, action);
        history.addTransition(currentState, action, reward);

        // sample a successor using the model for this trial
        State nextState = pModelThisTrial->sampleSuccessor(currentState, action);
        currentBelief = getNewBelief(currentBelief, currentState, action, nextState);
        currentState = nextState;
        depth++;
    }

    return history;
}

/* Make a copy of the current belief and return a pointer to a belief which is
updated according to the transition observed. */
std::shared_ptr<FiniteMDPBelief> BamdpCvarPGApprox::getNewBelief(
        std::shared_ptr<FiniteMDPBelief> currentBelief,
        State s,
        std::string action,
        State nextState
){

    // make a copy of the belief from this existing node
    std::shared_ptr<FiniteMDPBelief> newBelief(currentBelief->clone());

    // update the belief according to the transition observed
    newBelief->updateBelief(s, action, nextState);
    return newBelief;
}


/* Execute policy gradient updates */
void BamdpCvarPGApprox::runTrials(
    int batchSize,
    int iterations
){

    // get a template MDP to use for enabled actions and rewards
    std::shared_ptr<MDP> pTemplateMDP = rootSamples[0];
    std::unordered_map<State, int, StateHash> stateMapping = pTemplateMDP->getStateMapping();
    std::unordered_map<std::string, int> actionMapping = pTemplateMDP->getActionMapping();

    // random int generator to sample models from memory
    int numModels = rootSamples.size();
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, numModels-1);

    int it = 0;
    while(it < iterations){
        std::cout << "Iteration: " << it << ", ";

        // reset variables for this batch
        std::vector<std::vector<State>> trajectoryStates;
        std::vector<std::vector<std::string>> trajectoryActions;
        std::vector<std::vector<std::shared_ptr<FiniteMDPBelief>>> trajectoryBeliefs;
        std::vector<float> returns;
        int trial = 0;

        while(trial < batchSize){

            // sample the model for this trial to use root sampling
            int modelIndex = uni(rng);
            std::shared_ptr<MDP> pModelThisTrial = rootSamples.at(modelIndex);

            // reset variables for this trial
            float rewardThisEpisode = 0.0;
            int depth = 0;
            std::shared_ptr<FiniteMDPBelief> currentBelief = rootParticleBelief;
            State currentState = initState;
            std::vector<State> states;
            std::vector<std::string> actions;
            std::vector<std::shared_ptr<FiniteMDPBelief>> beliefs;


            while(depth < horizon){

                // select action and record reward received
                std::string action = selectActionSoftmax(
                        currentState,
                        currentBelief,
                        pTemplateMDP,
                        stateMapping,
                        actionMapping
                    );

                float reward = pTemplateMDP->getReward(currentState, action);
                states.push_back(currentState);
                actions.push_back(action);
                beliefs.push_back(currentBelief);
                rewardThisEpisode += reward;

                // sample a successor using the model for this trial
                State nextState = pModelThisTrial->sampleSuccessor(currentState, action);
                currentBelief = getNewBelief(currentBelief, currentState, action, nextState);
                currentState = nextState;
                depth++;
            }
            // store the result of this trial
            trajectoryStates.push_back(states);
            trajectoryActions.push_back(actions);
            trajectoryBeliefs.push_back(beliefs);
            returns.push_back(rewardThisEpisode);
            trial++;
        }

        gradientUpdate(
                trajectoryStates,
                trajectoryActions,
                trajectoryBeliefs,
                returns,
                pTemplateMDP,
                stateMapping,
                actionMapping
        );

        it++;
    }
}

/* perform a gradient update to all the parameters in the W matrix. */
void BamdpCvarPGApprox::gradientUpdate(
        std::vector<std::vector<State>>& trajectoryStates,
        std::vector<std::vector<std::string>>& trajectoryActions,
        std::vector<std::vector<std::shared_ptr<FiniteMDPBelief>>>& trajectoryBeliefs,
        std::vector<float>& returns,
        std::shared_ptr<MDP> pTemplateMDP,
        std::unordered_map<State, int, StateHash>& stateMapping,
        std::unordered_map<std::string, int>& actionMapping
){
    // compute the empirical estimate of the VaR
    std::vector<float> returnsSorted = returns;
    std::sort(returnsSorted.begin(), returnsSorted.end());
    int varIndex = (int)(std::round(returns.size() * alpha)) - 1;
    float varEstimate = returnsSorted[varIndex];
    float cvarSum = 0.0;
    for(int i = 0; i <= varIndex; i++){
        cvarSum += returnsSorted[i];
    }
    std::cout << "Batch empirical cvar: " << cvarSum/(varIndex + 1) << std::endl;

    std::vector<std::vector<float>> grad(mdpMapping.size(), std::vector<float>(actionMapping.size()*stateMapping.size(), 0.0));
    for(unsigned int i = 0; i < returns.size(); i++){
        if(returns[i] >  varEstimate){
            continue;
        }
        std::vector<State> states = trajectoryStates[i];
        std::vector<std::string> actions = trajectoryActions[i];
        std::vector<std::shared_ptr<FiniteMDPBelief>> beliefs = trajectoryBeliefs[i];
        float ret = returns[i];

        for(unsigned int j = 0; j < states.size(); j++){
            State s = states[j];
            std::string actionTaken = actions[j];
            std::shared_ptr<FiniteMDPBelief> b = beliefs[j];
            std::unordered_map<std::shared_ptr<MDP>, float> beliefWeights = b->getWeights();

            std::unordered_map<std::string, float> actionWeights;
            for(std::string act : pTemplateMDP->getEnabledActions(s)){
                int stateActionIndex = getStateActionIndex(
                    s,
                    act,
                    stateMapping,
                    actionMapping);

                float actionWeight = 0.0;
                for(auto pair : beliefWeights){
                    std::shared_ptr<MDP> pMDP = pair.first;
                    actionWeight += W[mdpMapping[pMDP]][stateActionIndex]*pair.second;
                }
                actionWeights[act] = actionWeight;
            }

            float expSum = 0.0;
            for(auto pair : actionWeights){
                expSum += std::exp(pair.second);
            }

            for(std::string act : pTemplateMDP->getEnabledActions(s)){
                int stateActionIndex = getStateActionIndex(
                    s,
                    act,
                    stateMapping,
                    actionMapping);

                float dJ_dweight;
                if(act == actionTaken){
                    dJ_dweight = (1 - std::exp(actionWeights[act])/expSum)*(ret - varEstimate)/alpha;
                }else{
                    dJ_dweight = -std::exp(actionWeights[act])/expSum*(ret - varEstimate)/alpha;
                }


                for(auto pair : beliefWeights){
                    float dweight_dtheta = pair.second;
                    float dJ_dtheta = dJ_dweight * dweight_dtheta;
                    grad[mdpMapping[pair.first]][stateActionIndex] += dJ_dtheta;
                }
            }
        }
    }

    for(auto m : mdpMapping){
        for(auto s : pTemplateMDP->enumerateStates()){
            for(auto action : actionMapping){
                int stateActionIndex = getStateActionIndex(
                    s,
                    action.first,
                    stateMapping,
                    actionMapping);

                W[m.second][stateActionIndex] += grad[m.second][stateActionIndex] * learningRate;
            }
        }
    }
}

/* initialise the weighting matrix based on the actions chosen by the cvar expected mdp
policy */
void BamdpCvarPGApprox::initMatrixWithCvarPolicy(
    std::shared_ptr<Belief> pBelief
){
    std::shared_ptr<MDP> pExpectedMDP = pBelief->getExpectedMDP();
    int numInterpPts = 10;
    std::cout << "Computing expected MDP cvar policy for initialisation..." << std::endl;
    CvarValueIteration solver(numInterpPts);
    solver.valueIteration(*pExpectedMDP);
    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);
    float initValue = 2.0;


    for(auto s : pExpectedMDP->enumerateStates()){
        std::string action;
        std::unordered_map<State, float, StateHash> perturbation;
        std::tie(action, perturbation) = solver.getOptimalAction(
            pExpectedMDP,
            s,
            alpha,
            env
        );

        std::unordered_map<std::string, int> actionMap = pExpectedMDP->getActionMapping();
        std::unordered_map<State, int, StateHash> stateMap = pExpectedMDP->getStateMapping();
        int stateActionIndex = getStateActionIndex(
            s,
            action,
            stateMap,
            actionMap
        );
        for(auto m : mdpMapping){
            W[m.second][stateActionIndex] = initValue;
        }
    }
    std::cout << "Done." << std::endl;
}


/* select an action according to the current parameter weights */
std::string BamdpCvarPGApprox::selectActionSoftmax(
    State currentState,
    std::shared_ptr<FiniteMDPBelief> currentBelief,
    std::shared_ptr<MDP> pTemplateMDP,
    std::unordered_map<State, int, StateHash>& stateMapping,
    std::unordered_map<std::string, int>& actionMapping
){
    std::unordered_map<std::string, float> actionWeights;
    for(std::string act : pTemplateMDP->getEnabledActions(currentState)){
        int stateActionIndex = getStateActionIndex(
            currentState,
            act,
            stateMapping,
            actionMapping);

        std::unordered_map<std::shared_ptr<MDP>, float> beliefWeights = currentBelief->getWeights();
        float actionWeight = 0.0;
        for(auto pair : beliefWeights){
            std::shared_ptr<MDP> pMDP = pair.first;
            actionWeight += W[mdpMapping[pMDP]][stateActionIndex]*pair.second;
        }
        actionWeights[act] = actionWeight;
    }

    std::vector<float> weights;
    float weightSum = 0.0;
    for(auto pair : actionWeights){
        weightSum += std::exp(pair.second);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::string> actions;
    std::vector<float> probs;
    actions.reserve(actionWeights.size());
    probs.reserve(actionWeights.size());

    for(auto kv : actionWeights) {
        actions.push_back(kv.first);
        probs.push_back(std::exp(kv.second)/weightSum);
    }

    // sample a state index according to perturbed distribution
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int ind = dist(gen);
    std::string action = actions.at(ind);
    return action;
}


int BamdpCvarPGApprox::getStateActionIndex(
        State s,
        std::string act,
        std::unordered_map<State, int, StateHash>& stateMapping,
        std::unordered_map<std::string, int>& actionMapping)
{
    return stateMapping.size() * actionMapping[act] + stateMapping[s];
}
