#include "multimodel_mdp.h"
#include <random>

MultiModelMDP::MultiModelMDP(std::unordered_map<std::shared_ptr<MDP>, float> weights_)
: weights(weights_)
{
}

std::shared_ptr<MDP> MultiModelMDP::sampleModel() {
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

std::unordered_map<std::shared_ptr<MDP>, float> MultiModelMDP::getWeights(){
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
void MultiModelMDP::updateBelief(State initState, std::string action, State nextState){
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
