#include "finite_mdp_belief.h"
#include "mdp.h"
#include <random>

FiniteMDPBelief::FiniteMDPBelief(std::unordered_map<std::shared_ptr<MDP>, float> weights_)
: weights(weights_)
{
}

std::shared_ptr<MDP> FiniteMDPBelief::sampleModel(){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::shared_ptr<MDP>> mdpPointers;
    std::vector<float> mdpWeights;

    for(auto kv : weights) {
        mdpPointers.push_back(kv.first);
        mdpWeights.push_back(kv.second);
    }

    // sample a state index according to distribution defined by MDP
    std::discrete_distribution<> dist(mdpWeights.begin(), mdpWeights.end());
    int mdpIndex = dist(gen);
    return mdpPointers.at(mdpIndex);
}

std::unordered_map<std::shared_ptr<MDP>, float> FiniteMDPBelief::getWeights() const {
    return weights;
}

/* updates the belief probabilities associated with each MDP sample after
observing a transition in the MDP. The function updates the weights attribute
of the class.

Args:
    initState: the initial state of the transition.
    action: the action applied in the transition.
    nextState: the next state observed in the transition.
*/
void FiniteMDPBelief::updateBelief(State initState, std::string action, State nextState){
    std::unordered_map<std::shared_ptr<MDP>, float> newWeights;
    float eta = 0.0;

    for(auto kv : weights){
        std::shared_ptr<MDP> pMDP = kv.first;
        float mdpProb = kv.second;
        std::unordered_map<State, float, StateHash> transProbs = pMDP->getTransitionProbs(initState, action);

        // if nextState is not a possible set weighting to zero
        if(transProbs.count(nextState) == 0){
            newWeights[pMDP] = 0.0;
            continue;

        // otherwise probability is proportional to probability of transition
        // times probability of MDP.
        }else{
            newWeights[pMDP] = mdpProb*transProbs[nextState];
            eta += mdpProb*transProbs[nextState];
        }
    }

    // divide each of the weights by normalisation constant to make probabilities.
    for(auto kv : newWeights){
        newWeights[kv.first] = kv.second/eta;
    }

    // set the weights to their new values
    weights = newWeights;
}

/* Return weights for a particle filter for this belief */
std::unordered_map<std::shared_ptr<MDP>, float> FiniteMDPBelief::getParticleWeights(int numParticles) {
    if((unsigned int)numParticles > weights.size()){
        throw "Particles greater than samples in belief";
    }

    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    int i = 0;
    for(auto kv : weights){
        std::shared_ptr<MDP> pMDP = kv.first;
        mdpWeights[pMDP] = 1.0/numParticles;
        if(i == numParticles-1){
            break;
        }
        i++;
    }
    return mdpWeights;
}

/* Return the transition probabilities for executing a state-action pair according
to the current belief distribution

Args:
    state: the state of interest.
    action: the action executed.

Returns:
    a mapping defining the resulting transition probabilities. */
std::unordered_map<State, float, StateHash> FiniteMDPBelief::getBeliefTransitionProbs(State state, std::string action) const{
    std::unordered_map<State, float, StateHash> beliefTransitionProbs;
    for(auto mdpWeightPair : weights){
        std::shared_ptr<MDP> pMDP = mdpWeightPair.first;
        float mdpProb = mdpWeightPair.second;
        std::unordered_map<State, float, StateHash> transProbs = pMDP->getTransitionProbs(state, action);

        for(auto transitionPair : transProbs){
            State nextState = transitionPair.first;

            // if this transition state is already in the output mapping add the probability mass
            if(beliefTransitionProbs.count(nextState) > 0){
                beliefTransitionProbs[nextState] += mdpProb * transitionPair.second;

            // otherwise create new entry
            }else{
                beliefTransitionProbs[nextState] = mdpProb * transitionPair.second;
            }
        }
    }

    return beliefTransitionProbs;
}


/* Return the expected MDP according to the current belief. This returns an
MDP which has transition probabilities which are a weighted average of the
transition probabilities in the MDP set, weighted according to the belief
probability. */
std::shared_ptr<MDP> FiniteMDPBelief::getExpectedMDP() const {
    std::shared_ptr<MDP> pMDP = weights.begin()->first;
    int nStates = pMDP->getNumStates();
    int nActions = pMDP->getNumActions();

    // initialise transition matrix of appt size
    std::vector<std::vector<std::vector<float>>> averageT(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));

    // loop through each mdp sample
    for(auto pair : weights){
        std::shared_ptr<MDP> pCurrentMDP = pair.first;
        float wt = pair.second;
        std::vector<std::vector<std::vector<float>>> currentMDPTransitionMat = pCurrentMDP->getTransitionMatrix();

        for(int i = 0; i < nStates; i++){
            for(int j = 0; j < nActions; j++){
                for(int k = 0; k < nStates; k++){
                    averageT[i][j][k] += currentMDPTransitionMat[i][j][k] * wt;
                }
            }
        }
    }

    std::shared_ptr<MDP> expectedMDP;
    expectedMDP = std::make_shared<MDP>(
                            pMDP->getInitialStateProbs(),
                            averageT,
                            pMDP->getRewardMatrix(),
                            pMDP->getStateMapping(),
                            pMDP->getActionMapping());
    return expectedMDP;
}
