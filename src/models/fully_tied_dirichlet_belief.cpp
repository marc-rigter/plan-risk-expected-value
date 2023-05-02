#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "fully_tied_dirichlet_belief.h"

/* constructor to implement belief based on Dirichlet count where the transition
probabilities are tied between different states and different actions:
the transition probabilities are the same regardless of the state or action.

Args:
    pMDP: this is an MDP model used only to define the actions and state space -
        not to compute transition probabilities.
    initialDirichlet_: the dirichlet disributions corresponding to the
        belief, one for each action
    pseudoStateMapping_: maps each state-action pair to a mapping between
        successor states and the "pseudo-state" string which corresponds to that
        successor state. T

To implement the parameter sharing, each action has a Dirichlet distribution
which contains successor probabilities to "psuedo states". The pseudoStateMapping
tells us how the pseudo states map onto real states according to the current
state-action pair. For example, the pseudo state may be the state which
corresponds to "veeringLeft", and we need the mapping to define which is the
state to the left. The probability for veering left is shared as the same across
all initial states.
*/
FullyTiedDirichletBelief::FullyTiedDirichletBelief(
    std::shared_ptr<MDP> pMDP_,
    TiedDirichletDistribution dirichletDistribution_,
    std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping_
) : pMDP(pMDP_), dirichletDistribution(dirichletDistribution_), pseudoStateMapping(pseudoStateMapping_){

}

void FullyTiedDirichletBelief::updateBelief(State initState, std::string action, State nextState){

    // find the pseudostate which corresponds to the state we have transitioned to
    successorMapping map = pseudoStateMapping->at(initState).at(action);
    std::string pseudoState;
    int count = 0;
    for(auto pair : map){
        if(pair.second == nextState){
            pseudoState = pair.first;
            count++;
        }
    }

    // if multiple pseudostates map to this successor state then this transition
    // should not be used to update the dirichlet belief
    if(count > 1){
        return;
    }

    // increment the dirichlet count for the successor pseudo state.
    dirichletDistribution.observe(pseudoState);
}

/* Return weights for a particle filter for this belief */
std::unordered_map<std::shared_ptr<MDP>, float> FullyTiedDirichletBelief::getParticleWeights(int numParticles) {
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    for(int n = 0; n < numParticles; n++){
        std::shared_ptr<MDP> pMDP = sampleModel();
        mdpWeights[pMDP] = 1.0/numParticles;
    }
    return mdpWeights;
}

/* Returns the expected transition probabilities according to the current
belief.

Args:
    state: the state to get transition probs.
    action: the action to get expected transition probs.
*/
std::unordered_map<State, float, StateHash> FullyTiedDirichletBelief::getBeliefTransitionProbs(State state, std::string action) const{
    std::unordered_map<std::string, float> dist = dirichletDistribution.getExpectedDistribution();
    successorMapping map = pseudoStateMapping->at(state).at(action);

    std::unordered_map<State, float, StateHash> beliefTransitionProbs;
    for(auto pair : map){
        float prob = dist.at(pair.first);
        beliefTransitionProbs[pair.second] += prob;
    }

    return beliefTransitionProbs;
}

/* Returns the transition probabilities according to a sample of the true
probability values from the Dirichlet distribution. */
std::unordered_map<State, float, StateHash> FullyTiedDirichletBelief::getSampleTransitionProbs(
        State state,
        std::string action,
        std::unordered_map<std::string, float> dist) const
{
    successorMapping map = pseudoStateMapping->at(state).at(action);

    std::unordered_map<State, float, StateHash> sampleTransitionProbs;
    for(auto pair : map){
        float prob = dist.at(pair.first);
        sampleTransitionProbs[pair.second] += prob;
    }

    return sampleTransitionProbs;
}

/* Returns the expected MDP which has transition probabilities according to
the expected values of the dirichlet distributions.

Args:
    None

Returns:
    a shared pointer to the expected MDP
*/
std::shared_ptr<MDP> FullyTiedDirichletBelief::getExpectedMDP() const{
    int nStates = pMDP->getNumStates();
    int nActions = pMDP->getNumActions();
    std::unordered_map<State, int, StateHash> stateMap = pMDP->getStateMapping();
    std::unordered_map<std::string, int> actionMap = pMDP->getActionMapping();

    // initialise transition matrix of appt size
    std::vector<std::vector<std::vector<float>>> beliefT(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));

    // loop through each mdp sample
    for(State s : pMDP->enumerateStates()){
        for(std::string act : pMDP->getEnabledActions(s)){
            std::unordered_map<State, float, StateHash> successorProbs = getBeliefTransitionProbs(s, act);

            for(auto pair : successorProbs){
                State nextS = pair.first;
                beliefT[stateMap[s]][actionMap[act]][stateMap[nextS]] = pair.second;
            }
        }
    }

    std::shared_ptr<MDP> expectedMDP;
    expectedMDP = std::make_shared<MDP>(
                            pMDP->getInitialStateProbs(),
                            beliefT,
                            pMDP->getRewardMatrix(),
                            stateMap,
                            actionMap);
    return expectedMDP;
}


/* Returns a sample of an MDP model according to the belief.

Returns:
    An MDP sample from the belief.
*/
std::shared_ptr<MDP> FullyTiedDirichletBelief::sampleModel() {
    int nStates = pMDP->getNumStates();
    int nActions = pMDP->getNumActions();
    std::unordered_map<State, int, StateHash> stateMap = pMDP->getStateMapping();
    std::unordered_map<std::string, int> actionMap = pMDP->getActionMapping();

    // initialise transition matrix of appt size
    std::vector<std::vector<std::vector<float>>> beliefT(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));

    // sample from the dirichlet distribution corresponding to the current belief
    std::unordered_map<std::string, float> dist = dirichletDistribution.sampleDistribution();

    // loop through each mdp sample
    for(State s : pMDP->enumerateStates()){
        for(std::string act : pMDP->getEnabledActions(s)){
            std::unordered_map<State, float, StateHash> successorProbs = getSampleTransitionProbs(s, act, dist);

            for(auto pair : successorProbs){
                State nextS = pair.first;
                beliefT[stateMap[s]][actionMap[act]][stateMap[nextS]] = pair.second;
            }
        }
    }

    std::shared_ptr<MDP> expectedMDP;
    expectedMDP = std::make_shared<MDP>(
                            pMDP->getInitialStateProbs(),
                            beliefT,
                            pMDP->getRewardMatrix(),
                            stateMap,
                            actionMap);
    return expectedMDP;
}

TiedDirichletDistribution FullyTiedDirichletBelief::getDirichletDistribution() const{
    return dirichletDistribution;
}
