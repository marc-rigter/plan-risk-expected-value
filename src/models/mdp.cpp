#include <iostream>
#include <cmath>
#include <unordered_map>
#include <random>
#include <numeric>
#include "state.h"
#include "mdp.h"
#include "utils.h"

/* initialise MDP by setting attributes and checking that the MDP parameters
are valid.

Args:
    initialStateProbs: mapping from states to probabilities defining initial
        state distribution.
    T_: 3 dimensional vector defining transition function
    R_: 2 dimensional vector defining state action rewards
    stateMapping_: a mapping from states to indices in each of the matrices.
    actionMapping_: a mapping from action strings to indices in the matrices.
*/
MDP::MDP(
    std::unordered_map<State, float, StateHash> initialStateProbs_,
    std::vector<std::vector<std::vector<float>>> T_,
    std::vector<std::vector<float>> R_,
    std::unordered_map<State, int, StateHash> stateMapping_,
    std::unordered_map<std::string, int> actionMapping_,
    bool check,
    transition_map transitionProbs_
)
: initialStateProbs{initialStateProbs_}, T{std::move(T_)}, transitionProbs(std::move(transitionProbs_)), R{std::move(R_)}, stateMapping{stateMapping_}, actionMapping{actionMapping_} {

    // initialise the inverse mappings from indices to states and actions
    int i = 0;
    for(auto const &pair: stateMapping){
        this->stateList.push_back(pair.first);
        this->inverseStateMapping[pair.second] = pair.first;
        i++;
    }

    for(auto const &pair: actionMapping){
        this->inverseActionMapping[pair.second] = pair.first;
    }

    numStates = stateMapping_.size();
    numActions = actionMapping_.size();
    if(check){
        this->checkMDP();
    }
}

std::unordered_map<State, float, StateHash> MDP::getInitialState() const{
    return this->initialStateProbs;
}

std::vector<State> MDP::enumerateStates() const{
    return this->stateList;
}

float MDP::getReward(State state, std::string action) const{
    return this->R.at(stateMapping.at(state)).at(actionMapping.at(action));
}

std::vector<std::string> MDP::getEnabledActions(State s) {
    if (enabledActionMapping.find(s) == enabledActionMapping.end()){
        enabledActionMapping[s] = computeEnabledActions(s);
    }
    return enabledActionMapping.at(s);
}

std::unordered_map<State, float, StateHash> MDP::getTransitionProbs(State state, std::string action) {
    if (transitionProbs.find(state) != transitionProbs.end()){

        // if state exists in hash table but not action action
        if(transitionProbs.at(state).find(action) == transitionProbs.at(state).end()){
            transitionProbs.at(state)[action] = computeTransitionProbs(state, action);
        }

    // if neither state nor action exist in hash table
    }else{
        std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
        stateTransitionProbs[action] = computeTransitionProbs(state, action);
        transitionProbs[state] = stateTransitionProbs;
    }

    return transitionProbs.at(state).at(action);
}

/* Retrieves the transition probabilities assocaited with a state action pair.

Args:
    state: the state of interest.
    action: action string of interest.

Returns:
    mapping from states to probabilities for states with a nonzero chance of
    being a successor.
*/
std::unordered_map<State, float, StateHash> MDP::computeTransitionProbs(State state, std::string action) const{
    std::unordered_map<State, float, StateHash> transitionProbs;
    float prob;

    for(int i = 0; i < numStates; i++){
        prob = this->T.at(stateMapping.at(state)).at(actionMapping.at(action)).at(i);

        if(prob > 0.0){
            transitionProbs[inverseStateMapping.at(i)] = prob;
        }
    }
    return transitionProbs;
}

/* Returns a successor state sampled according to the transition probabilities
from applying a state action pair

Args:
    s: the state to be applied
    action: the action to be applied

Returns:
    the randomly sampled succesor state
*/
State MDP::sampleSuccessor(State s, std::string action) {
    std::unordered_map<State, float, StateHash> transProbs = getTransitionProbs(s, action);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<State> states;
    std::vector<float> probs;
    states.reserve(transProbs.size());
    probs.reserve(transProbs.size());

    for(auto kv : transProbs) {
        states.push_back(kv.first);
        probs.push_back(kv.second);
    }

    // if there are no successors return same state
    if(states.size() == 0){
        return s;
    }

    // sample a state index according to distribution defined by MDP
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int stateIndex = dist(gen);
    State nextState = states.at(stateIndex);
    return nextState;
}

/* Returns a successor state sampled according to the transition probabilities
whcih are perturbed by the mapping provided. TODO: factor out code overlap with
standard sampling

Args:
    s: the state to be applied
    action: the action to be applied

Returns:
    the randomly sampled succesor state
*/
State MDP::samplePerturbedSuccessor(State s, std::string action, std::unordered_map<State, float, StateHash> perturbation) {
    std::unordered_map<State, float, StateHash> transProbs = getTransitionProbs(s, action);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<State> states;
    std::vector<float> probs;
    states.reserve(transProbs.size());
    probs.reserve(transProbs.size());

    for(auto kv : transProbs) {
        states.push_back(kv.first);

        // multiply the original transition probability by the perturbation
        float perturbedProb = kv.second*perturbation[kv.first];
        if(perturbedProb > 1.0){
            std::cerr << "Error: Invalid perturbed transition probability." << std::endl;
            std::exit(-1);
        }
        probs.push_back(perturbedProb);
    }

    if(!cmpf(accumulate(probs.begin(), probs.end(), 0.0f), 1.0)){
        std::cerr << "Error: Perturbed MDP probabilities not valid distribution" << std::endl;
        throw "Error: Perturbed MDP probabilities not valid distribution";
    }

    // if there are no successors return same state
    if(states.size() == 0){
        return s;
    }

    // sample a state index according to perturbed distribution
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int stateIndex = dist(gen);
    State nextState = states.at(stateIndex);
    return nextState;
}

/* Returns a random action sampled uniformly from the actions enabled at a
state.

Args:
    s: the state at which to sample a random action.

Returns:
    action defined by a string.
*/
std::string MDP::sampleAction(State s) {
    std::vector<std::string> actions = getEnabledActions(s);
    int numActions = actions.size();
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, numActions-1);
    int ind = uni(rng);

    return actions[ind];
}


/* returns if a state is a goal state containing only one action which is a
self loop */
bool MDP::isGoal(State s){
    std::vector<std::string> enabledActions = getEnabledActions(s);
    if(enabledActions.size() != 1){
        return false;
    }

    // state is goal if there is only one action one successor and the successor
    // is the same state
    std::unordered_map<State, float, StateHash> probs = getTransitionProbs(s, enabledActions[0]);
    if(probs.size() == 1 && probs.begin()->first == s){
        return true;
    }else{
        return false;
    }
}

/* computes enabled actions at each state. These are actions for
which the sum over transition probabilities for executing the state action pair
is nonzero.
*/
std::vector<std::string> MDP::computeEnabledActions(State s) const{
    std::vector<std::string> actions;

    // no transition matrix use hash table
    if(T.size() < 1){
        for(auto kv : transitionProbs.at(s)){
            actions.push_back(kv.first);
        }
        return actions;
    }

    int i = stateMapping.at(s);

    for(int j = 0; j < numActions; j++){
        float probSum = 0.0;
       for(int k = 0; k < numStates; k++){
            probSum += T[i][j][k];
        }
        if(probSum > 0.0){
            actions.push_back(inverseActionMapping.at(j));
        }
    }
    return actions;
}

/* Checks the validity of the MDP by: checking the dimensions of the matrices
provided, checking for a valid initial state distribution, and checking
for a valid transition matrix.
*/
void MDP::checkMDP(){

    // check size of transition matrix
    if(T.size() != stateMapping.size()){
        std::cout << T.size() << std::endl << stateMapping.size();
        std::cerr << "Error: Transition matrix has incorrect dimensions" << std::endl;
        std::exit(-1);
    }

    for(unsigned int i = 0; i < T.size(); i++){
        for(unsigned int j = 0; j < T[i].size(); j++){
            if(T[i].size() != actionMapping.size() || T[i][j].size() != stateMapping.size()){
                std::cerr << "Error: Transition matrix has incorrect dimensions" << std::endl;
                std::exit(-1);
            }
        }
    }

    // check size of reward matrix
    if(R.size() != stateMapping.size()){
        std::cerr << "Error: Reward matrix has incorrect dimensions" << std::endl;
        std::exit(-1);
    }

    for(unsigned int i = 0; i < R.size(); i++){
        if(R[i].size() != actionMapping.size()){
            std::cerr << "Error: Reward matrix has incorrect dimensions" << std::endl;
            std::exit(-1);
        }
    }

    // initial state transition probabilities must sum to 1
    float probSum = 0.0;
    for(std::pair<State, float> pair : this->initialStateProbs){
        probSum += pair.second;
    }

    if(!cmpf(probSum, 1.0)){
        std::cerr << "Error: Invalid initial state distribution" << std::endl;
        std::exit(-1);
    }

    // check that the transition probabilities for any state action pair
    // sum to either 0 for disabled actions or 1 for enabled actions
    for(int i = 0; i < numStates; i++){
        for(int j = 0; j < numActions; j++){
            probSum = 0.0;

            for(int k = 0; k < numStates; k++){
                probSum += T[i][j][k];
            }
            if(!(cmpf(probSum, 1.0) || cmpf(probSum, 0.0))){
                std::cout << "state: " << inverseStateMapping[i] << std::endl;
                std::cout << "action: " << inverseActionMapping[j] << std::endl;
                std::cerr << "Error: Invalid transition matrix" << std::endl;
                std::exit(-1);
            }
        }
    }
}
