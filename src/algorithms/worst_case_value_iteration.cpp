#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"
#include "worst_case_value_iteration.h"

/* returns the Q value of a state action pair given a current value function
estimate. for worst-case value iteration which computes the worst-case value
we assume that the transition to the worst possible value occurs.

Args:
    m: an MDP.
    action: the action to compute the Q value for.
    value: the value function.
*/
float worstCaseVI::getQVal(MDP& m, State s, std::string action, std::unordered_map<State, float, StateHash> value, bool maximise){
    float worstQVal;
    float reward = m.getReward(s, action);
    std::unordered_map<State, float, StateHash> tranProbs;

    if(maximise){
        worstQVal = std::numeric_limits<float>::max();
    }else{
        worstQVal = -std::numeric_limits<float>::max();
    }

    tranProbs = m.getTransitionProbs(s, action);
    for(std::pair<State, float> pair : tranProbs){

        // assume transition to worst successor rather than probabilistic
        float qVal = reward + value[pair.first];

        if(maximise){
            if(qVal < worstQVal){
                worstQVal = qVal;
            }
        }else{
            if(qVal > worstQVal){
                worstQVal = qVal;
            }
        }
    }

    return worstQVal;
}
