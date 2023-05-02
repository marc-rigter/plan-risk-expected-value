#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <queue>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "gurobi_c++.h"
#include "cvar_value_iteration.h"
#include "cvar_lexicographic.h"

std::tuple<std::string, std::string> getNearestCost(
    std::string cost,
    std::vector<float> costInterpFloats,
    std::vector<std::string> costInterpPts
){

    if(std::stof(cost) >= std::stof(costInterpPts.back())){
      return std::make_tuple(costInterpPts.back(), costInterpPts.back());
    }

    if(std::stof(cost) <= std::stof(costInterpPts.front())){
      return std::make_tuple(costInterpPts.front(), costInterpPts.front());
    }

    auto const it = std::upper_bound(costInterpFloats.begin(), costInterpFloats.end(), std::stof(cost));
    std::string costAbove = std::to_string(*it).substr(0, 7);
    std::string costBelow = std::to_string(*std::prev(it)).substr(0, 7);

    return std::make_tuple(costBelow, costAbove);
}

float CvarLexicographic::interpCostValue(
    std::unordered_map<State, float, StateHash>& value,
    State augmentedState
){
    std::string lowerCost;
    std::string upperCost;
    std::tie(lowerCost, upperCost) = getNearestCost(augmentedState.getValue("cost_so_far"), costInterpFloats, costInterpPts);
    float currentCost = std::stof(augmentedState.getValue("cost_so_far"));

    std::unordered_map<std::string, std::string> stateMap = augmentedState.getStateMapping();
    stateMap["cost_so_far"] = lowerCost;
    State stateCostBelow(stateMap);
    stateMap["cost_so_far"] = upperCost;
    State stateCostAbove(stateMap);

    if(currentCost > std::stof(upperCost) || cmpf(std::stof(upperCost), currentCost, 1e-4)){
        return value[stateCostAbove];
    }

    if(cmpf(std::stof(lowerCost), currentCost, 1e-4)){
        return value[stateCostBelow];
    }

    // linear interpolation
    float interp = value[stateCostBelow]*(std::stof(upperCost) - currentCost);
    interp += value[stateCostAbove]*(currentCost - std::stof(lowerCost));
    interp /= (std::stof(upperCost) - std::stof(lowerCost));

    return interp;
}

CvarLexicographic::CvarLexicographic(
    int numInterpPts_,
    float VaR_,
    std::unordered_map<State, float, StateHash> worstCaseValue_,
    std::unordered_map<State, std::string, StateHash> worstCasePolicy_
) : VaR(VaR_), worstCaseValue(worstCaseValue_), worstCasePolicy(worstCasePolicy_)
{
    float val = 0.0;
    for(int i = 0; i <= numInterpPts_; i++){
        costInterpPts.push_back(std::to_string(val).substr(0, 7));
        costInterpFloats.push_back(val);
        val += VaR_/numInterpPts_;
    }
}

/* gets the policy optimising expected value which guarantees to stay above a VaR
threshold. Assumes that cost is to be minimised. */
std::unordered_map<State, float, StateHash> CvarLexicographic::computeLexicographicValue(
    std::shared_ptr<MDP> pMDP,
    bool isSSP
){
    std::unordered_map<State, std::vector<std::string>, StateHash> prunedActions = getPrunedActions(
        pMDP
    );

    value = approximateVI(pMDP, prunedActions, isSSP);
    return value;
}

float CvarLexicographic::getQValue(
    std::shared_ptr<MDP> pMDP,
    State baseState,
    float costSoFar,
    std::string action
){
    std::unordered_map<State, float, StateHash> transitionProbs = pMDP->getTransitionProbs(baseState, action);
    std::unordered_map<std::string, std::string> stateMap = baseState.getStateMapping();
    float costAfterAction = pMDP->getReward(baseState, action) + costSoFar;

    float qVal = pMDP->getReward(baseState, action);
    for(auto pair : transitionProbs){
        State nextState = pair.first;
        std::unordered_map<std::string, std::string> mapping = nextState.getStateMapping();
        mapping["cost_so_far"] = std::to_string(costAfterAction);
        State nextAugmentedState = State(mapping);
        qVal += pair.second * interpCostValue(value, nextAugmentedState);
    }
    return qVal;
}

/* return the optimal expected value action which guarantees remaining below the VaR
threshold*/
std::string CvarLexicographic::getOptimalAction(
    std::shared_ptr<MDP> pMDP,
    State baseState,
    float costSoFar
){
    std::string bestAction;
    float bestValue = std::numeric_limits<float>::max();

    // only choose between actions guaranteed to maintain better than VaR
    std::vector<std::string> allowedActions = getAllowedActions(pMDP, baseState, costSoFar);
    for(std::string action : allowedActions){

        float qVal = getQValue(
            pMDP,
            baseState,
            costSoFar,
            action
        );

        if(qVal < bestValue){
            bestValue = qVal;
            bestAction = action;
        }
    }
    return bestAction;
}

/* performs approximate value iteration using the pruned allowed actions
 using interpolation */
std::unordered_map<State, float, StateHash> CvarLexicographic::approximateVI(
    std::shared_ptr<MDP> pMDP,
    std::unordered_map<State, std::vector<std::string>, StateHash> prunedActions,
    bool isSSP
)
{
    std::unordered_map<State, float, StateHash> val;
    for(auto pair : prunedActions){
        val[pair.first] = 0.0;
    }

    // not finite horizon
    if(isSSP){
        float maxError = std::numeric_limits<float>::max();
        int iter = 0;
        while(maxError > 1e-3){
            maxError = 0.0;

            // loop through each state which is stored in dict
            for(auto pair : prunedActions){
                State s = pair.first;
                float valueOld = val[s];
                approximateVIBackup(s, pMDP, val, prunedActions);
                float error = std::fabs(val[s] - valueOld);
                if(error > maxError){
                    maxError = error;
                }
            }
            std::cout << "error: " << maxError << std::endl;
            iter++;
        }

    // finite horizon
    }else{
        int maxT = 0;
        for(auto pair : prunedActions){
            if(std::stoi(pair.first.getValue("t")) > maxT){
                maxT = std::stoi(pair.first.getValue("t"));
            }
        }

        for(int t = maxT - 1; t >= 0; t--){
            std::cout << "t: " << t << std::endl;
            for(auto pair : prunedActions){
                State s = pair.first;
                if(std::stoi(s.getValue("t")) == t){
                    approximateVIBackup(s, pMDP, val, prunedActions);
                }
            }
        }
    }


    return val;
}

void CvarLexicographic::approximateVIBackup(
    State augmentedState,
    std::shared_ptr<MDP> pMDP,
    std::unordered_map<State, float, StateHash>& val,
    std::unordered_map<State, std::vector<std::string>, StateHash>& prunedActions
){
    float bestValue = std::numeric_limits<float>::max();

    // only allow the actions which are not pruned to be considered
    for(std::string action : prunedActions[augmentedState]){
        std::unordered_map<std::string, std::string> stateMap = augmentedState.getStateMapping();
        stateMap.erase("cost_so_far");
        State baseState(stateMap);
        float qVal = pMDP->getReward(baseState, action);

        float costAfterAction = pMDP->getReward(baseState, action) + std::stof(augmentedState.getValue("cost_so_far"));

        std::unordered_map<State, float, StateHash> transitionProbs = pMDP->getTransitionProbs(baseState, action);
        for(auto pair : transitionProbs){
            State nextState = pair.first;
            std::unordered_map<std::string, std::string> mapping = nextState.getStateMapping();
            mapping["cost_so_far"] = std::to_string(costAfterAction);
            State nextAugmentedState = State(mapping);
            qVal += pair.second * interpCostValue(val, nextAugmentedState);

        }
        if(qVal < bestValue){
            bestValue = qVal;
        }
    }

    val[augmentedState] = bestValue;
}

/* computes an MDP in which the only enabled actions are those in the worst-case
policy and those actions which are guaranteed to keep the total return better
than the VaR threshold. Thus policies defined on this MDP are guaranteed to
keep the return better than the VaR threshold.

Assumes that cost is to be minimised by keeping below VaR threshold.
 */
std::unordered_map<State, std::vector<std::string>, StateHash> CvarLexicographic::getPrunedActions(
    std::shared_ptr<MDP> pMDP
){

    std::unordered_map<State, std::vector<std::string>, StateHash> prunedActions;
    std::unordered_map<std::string, std::string> stateMap;
    for(auto baseState : pMDP->enumerateStates()){
        for(auto costVal : costInterpPts){
            stateMap = baseState.getStateMapping();
            stateMap["cost_so_far"] = costVal;
            State augState(stateMap);
            std::vector<std::string> allowedActions = getAllowedActions(pMDP, baseState, std::stof(costVal));
            prunedActions[augState] = allowedActions;
        }
    }

    return prunedActions;
}

std::vector<std::string> CvarLexicographic::getAllowedActions(
    std::shared_ptr<MDP> pMDP,
    State baseState,
    float costSoFar
){
    std::vector<std::string> allowedActions;
    for(auto act : pMDP->getEnabledActions(baseState)){

        // allow the action in the policy optimising worst return
        if(act == worstCasePolicy[baseState]){
            allowedActions.push_back(act);
            continue;
        }

        float worstReturnVal = -std::numeric_limits<float>::max();
        for(auto pair : pMDP->getTransitionProbs(baseState, act)){
            float returnVal  = pMDP->getReward(baseState, act) + worstCaseValue[pair.first];
            if(returnVal > worstReturnVal){
                worstReturnVal = returnVal;
            }
        }

        // if the cost so far plus the worst return from executing this
        // action is less then the VaR this action is enabled
        if(costSoFar + worstReturnVal <= VaR){
            allowedActions.push_back(act);
        }
    }
    return allowedActions;
}
