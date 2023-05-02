#ifndef mdp
#define mdp
#include <string>
#include "state.h"
#include <unordered_map>

/* Implements an MDP represented using matrices for transition functions and
rewards.

Attributes:
    initialStateProbs: mapping of states to initial probabilities.
    T: 3 dimensional vector defining transition probabilties
    R: 2 dimensiaonl vector defining rewards.
    stateMapping: mapping from states to index within the matrices.
    inverseStateMapping: mapping from indices to states.
    actionMapping: mapping from action strings to index within the matrices.
    inverseActionMapping: mapping from indices to action strings.
    enabledActionMapping: mapping from states to list of actions available at
        each state.
    stateList: list of states in the MDP.
    numStates: number of states in the MDP.
    numActions: number of actions in the MDP.
*/
class MDP {
private:
    typedef std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transition_map;
    std::unordered_map<State, float, StateHash> initialStateProbs;
    std::vector<std::vector<std::vector<float>>> T;
    transition_map transitionProbs;
    std::vector<std::vector<float>> R;
    std::unordered_map<State, int, StateHash> stateMapping;
    std::unordered_map<int, State> inverseStateMapping;
    std::unordered_map<std::string, int> actionMapping;
    std::unordered_map<int, std::string> inverseActionMapping;
    std::unordered_map<State, std::vector<std::string>, StateHash> enabledActionMapping;
    std::vector<State> stateList;
    int numStates;
    int numActions;
    bool asMatrix;

    void checkMDP();
    void checkTDict();
    std::vector<std::string> computeEnabledActions(State s) const;
    std::unordered_map<State, float, StateHash> computeTransitionProbs(State state, std::string action) const;

public:


    MDP(
        std::unordered_map<State, float, StateHash> initialStateProbs_,
        std::vector<std::vector<std::vector<float>>> T_,
        std::vector<std::vector<float>> R_,
        std::unordered_map<State, int, StateHash> stateMapping_,
        std::unordered_map<std::string, int> actionMapping_,
        bool check_=true,
        transition_map transitionProbs_=transition_map()
    );

    MDP(){};

    std::unordered_map<State, float, StateHash> getInitialState() const;
    std::vector<State> enumerateStates() const;
    std::unordered_map<State, float, StateHash> getTransitionProbs(State state, std::string action);
    float getReward(State state, std::string action) const;
    std::vector<std::string> getEnabledActions(State s);
    State sampleSuccessor(State s, std::string action);
    State samplePerturbedSuccessor(State s, std::string action, std::unordered_map<State, float, StateHash> perturbation);
    std::string sampleAction(State s);
    std::unordered_map<State, float, StateHash> getInitialStateProbs() const {return initialStateProbs;}
    std::vector<std::vector<std::vector<float>>> getTransitionMatrix() const {return T;}
    std::vector<std::vector<float>> getRewardMatrix() const {return R;}
    std::unordered_map<State, int, StateHash> getStateMapping() const {return stateMapping;}
    std::unordered_map<std::string, int> getActionMapping() const {return actionMapping;}
    int getNumStates() const {return numStates;}
    int getNumActions() const {return numActions;}
    bool isGoal(State s);
};

#endif
