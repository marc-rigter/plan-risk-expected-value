#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"

/* returns the Q value of a state action pair given a current value function
estimate.

Args:
    m: an MDP.
    action: the action to compute the Q value for.
    value: the value function.
*/
float VI::getQVal(MDP& m, State s, std::string action, std::unordered_map<State, float, StateHash> value, bool maximise){
    float qVal = m.getReward(s, action);
    std::unordered_map<State, float, StateHash> tranProbs;
    tranProbs = m.getTransitionProbs(s, action);

    for(std::pair<State, float> pair : tranProbs){
        qVal += pair.second * value[pair.first];
    }

    return qVal;
}

/* perform a bellman update at a state.

Args:
    m: the mdp defining the transition probabiltiies.
    s: the state to perform the bellman update at.
    maxReward: boolean which is true if the aim is to maaximise reward and
        false otherwise.
    value: the current value estimate as a mapping from states to floats.

*/
std::tuple<float, std::string> VI::bellmanUpdate(
                                    MDP& m,
                                    State s,
                                    bool maxReward,
                                    std::unordered_map<State, float, StateHash>& value
                                )
{
    std::vector<std::string> enabledActions = m.getEnabledActions(s);
    std::string updatedAction = "";
    float updatedVal;
    float qVal;

    // initialise to worst-case values
    if(maxReward){
        updatedVal = -std::numeric_limits<float>::max();
    }else{
        updatedVal = std::numeric_limits<float>::max();
    }

    // check each of the actions for the best q value
    bool update;
    for(auto action : enabledActions){
        qVal = getQVal(m, s, action, value, maxReward);
        update = false;


        if(maxReward && qVal > updatedVal){
            update = true;
        }else if(!maxReward && qVal < updatedVal){
            update = true;
        }

        if(update){
            updatedVal = qVal;
            updatedAction = action;
        }
    }

    return std::make_tuple(updatedVal, updatedAction);
}

/* Performs value iteration on the MDP provided.

Args:
    m: the MDP to perform value iteration upon.
    maxReward: boolean which is true if the aim is to maximise the reward.

Returns:
    tuple containing the value as a map from states to floats and the optimal
    policy as a mapping from states to strings.
*/
std::tuple<std::unordered_map<State, float, StateHash>, std::unordered_map<State, std::string, StateHash>> VI::valueIteration(MDP& m, bool maxReward){
    std::vector<State> allStates = m.enumerateStates();
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;
    float maxRes = -std::numeric_limits<float>::max();
    bool converged = false;

    for(auto s : allStates){
        value[s] = 0.0;
    }

    float newValue;
    std::string newAction;
    float res;

    while(!converged){
        for(auto s : allStates){
            std::tie(newValue, newAction) = bellmanUpdate(m, s, maxReward, value);
            policy[s] = newAction;
            res = fabs(newValue - value[s]);
            value[s] = newValue;

            maxRes = std::max(res, maxRes);
        }

        if(maxRes < 1e-5){
            converged = true;
        }

        std::cout << maxRes << std::endl;
        maxRes = -std::numeric_limits<float>::max();
    }

    return std::make_tuple(value, policy);
}
