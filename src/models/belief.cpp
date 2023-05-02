#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "hist.h"

/* convert the belief to a Bamdp by enumerating reachable beliefs over a horizon
from an initial state.

Args:
    pMDP: an MDP used only to find the enabled actions.
    initState: the initial state from which to enumerate reachable hyper states.
    horizon: the horizon over which to enumerate possible belief states.
*/
std::shared_ptr<MDP> Belief::toBamdp(std::shared_ptr<MDP> pMDP, State initState, int horizon){
    typedef std::tuple<State, State, std::shared_ptr<Belief>> Node;
    std::deque<Node> queue;
    std::vector<Node> nodeList;
    std::unordered_map<State, float, StateHash> initProbs;

    // add the initial state to the queue
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["history"] = initState.toString();
    stateMap["t"] = "0";
    State historyState(stateMap);
    std::shared_ptr<Belief> b(this->clone());
    queue.push_back(std::make_tuple(initState, historyState, b));
    initProbs[historyState] = 1.0;

    // enumerate the reachable belief states from the initial state
    State currentState;
    std::shared_ptr<Belief> currentBelief;
    while(queue.size() > 0){
        Node n = queue[0];
        currentState = std::get<0>(n);
        historyState = std::get<1>(n);
        currentBelief = std::get<2>(n);

        // add the state to the list
        nodeList.push_back(n);

        queue.pop_front();
        if(std::stoi(historyState.getValue("t")) < horizon){
            for(std::string act : pMDP->getEnabledActions(currentState)){
                std::unordered_map<State, float, StateHash> transProbs = currentBelief->getBeliefTransitionProbs(
                                                                                currentState,
                                                                                act);
                for(auto pair : transProbs){
                    State nextState = pair.first;
                    std::shared_ptr<Belief> newBelief(currentBelief->clone());
                    newBelief->updateBelief(currentState, act, nextState);

                    stateMap["history"] = getNewHistory(historyState, act, nextState);
                    stateMap["t"] = std::to_string(std::stoi(historyState.getValue("t"))+1);
                    State nextHistoryState(stateMap);
                    queue.push_back(std::make_tuple(nextState, nextHistoryState, newBelief));
                }
            }
        }
    }

    int nStates = nodeList.size();
    int nActions = pMDP->getNumActions();

    // initialise transition matrix of appt size
    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> bamdpTransitionProbs;
    std::vector<std::vector<std::vector<float>>> T;
    std::vector<std::vector<float>> bamdpR(nStates, std::vector<float>(nActions, 0.0));

    // generate state mapping
    int stateIndex = 0;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    for(auto n : nodeList){
        historyState = std::get<1>(n);
        mdpStateMap[historyState] = stateIndex;
        stateIndex++;
    }
    std::unordered_map<std::string, int> mdpActionMap = pMDP->getActionMapping();

    // for each of the histories add transition probs to matrix
    for(auto n : nodeList){
        currentState = std::get<0>(n);
        historyState = std::get<1>(n);
        currentBelief = std::get<2>(n);

        for(std::string act : pMDP->getEnabledActions(currentState)){
            std::unordered_map<State, float, StateHash> beliefProbs = currentBelief->getBeliefTransitionProbs(currentState, act);
            bamdpR[mdpStateMap[historyState]][mdpActionMap[act]] = pMDP->getReward(currentState, act);

            // at the maximum horizon value add a self loop so VI converges
            std::unordered_map<State, float, StateHash> successorProbs;
            if(std::stoi(historyState.getValue("t")) == horizon){
                successorProbs[historyState] = 1.0;
            }else{
                for(auto pair : beliefProbs){
                    State nextState = pair.first;

                    stateMap["history"] = getNewHistory(historyState, act, nextState);
                    stateMap["t"] = std::to_string(std::stoi(historyState.getValue("t"))+1);
                    State nextHistoryState(stateMap);
                    successorProbs[nextHistoryState] = pair.second;
                }
            }

            // if the state is in hash table
            if (bamdpTransitionProbs.find(historyState) != bamdpTransitionProbs.end()){

                // if state exists in hash table but not action action
                if(bamdpTransitionProbs.at(historyState).find(act) == bamdpTransitionProbs.at(historyState).end()){
                    bamdpTransitionProbs.at(historyState)[act] = successorProbs;
                }

            // if neither state nor action exist in hash table
            }else{
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs[act] = successorProbs;
                bamdpTransitionProbs[historyState] = stateTransitionProbs;
            }
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, bamdpR, mdpStateMap, mdpActionMap, check, bamdpTransitionProbs);
}

/* transProbs are an optional argument for the unperturbed transition probabilities
for the state-action pair under this belief. If the argument is provided, we
avoid recomputing those transition probabilities */
State Belief::sampleSuccessor(
    State s,
    std::string action,
    std::unordered_map<State, float, StateHash> transProbs) const
{

    // if the transition probabilities are not supplied they must be computed
    if(transProbs.size() == 0){
        transProbs = getBeliefTransitionProbs(s, action);
    }

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

/* transProbs are an optional argument for the unperturbed transition probabilities
for the state-action pair under this belief. If the argument is provided, we
avoid recomputing those transition probabilities */
State Belief::samplePerturbedSuccessor(
    State s,
    std::string action,
    std::unordered_map<State, float, StateHash> perturbation,
    std::unordered_map<State, float, StateHash> transProbs) const
{
    // if the transition probabilities are not supplied they must be computed
    if(transProbs.size() == 0){
        transProbs = getBeliefTransitionProbs(s, action);
    }

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
        float eps = 1e-6;
        if(perturbedProb > 1.0 + eps){
            std::cerr << "Error: Invalid belief perturbed transition probability." << std::endl;
            std::exit(-1);
        }
        probs.push_back(perturbedProb);
    }

    if(!cmpf(accumulate(probs.begin(), probs.end(), 0.0f), 1.0)){
        std::cerr << "Error: Perturbed belief probabilities not valid distribution" << std::endl;
        throw "Error: Perturbed belief probabilities not valid distribution";
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
