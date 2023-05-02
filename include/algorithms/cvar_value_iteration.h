#ifndef cvar_value_iteration
#define cvar_value_iteration
#include <string>
#include "state.h"
#include "mdp.h"
#include "gurobi_c++.h"
#include "cvar_hist.h"
#include "cvar_lexicographic.h"

/* Class to implement the Bayes adaptive Monte Carlo search approach to solving
Bayes adaptive MDPs.

Attributes:
    maxTrials: the maximum number of trials to be used to find the next action.
*/
class CvarValueIteration {
private:
    std::vector<float> alphaVals;
    std::vector<float> varAlphaVals;
    bool valueComputed;
    int maxT;
    std::unordered_map<State, float, StateHash> cvarValueFunction;
    std::vector<float> getAlphaValues(int numPts);
    State augToNormalState(State augState);
    std::unordered_map<State, std::map<std::string, std::string>, StateHash> cvarPolicy;
    std::unordered_map<State, std::map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> cvarPerturbationPolicy;

    void cvarBackup(
        State currentAugState,
        std::unordered_map<State, float, StateHash>& cvarValue,
        GRBEnv env,
        MDP& m,
        bool maximise=true
    );

    void varBackup(
        State currentAugState,
        std::unordered_map<State, float, StateHash>& varValue,
        GRBEnv env,
        std::shared_ptr<MDP> m,
        bool maximise=true
    );

    GRBModel getCvarLP(
        State currentState,
        float currentAlpha,
        std::string act,
        std::unordered_map<State, float, StateHash>& cvarValue,
        GRBEnv& env,
        MDP& m,
        bool maximise=true
    );

public:
    CvarValueIteration(int numPts_);
    std::unordered_map<State, float, StateHash> sspValueIteration(MDP& m, bool maximise=true); // for ssp
    std::unordered_map<State, float, StateHash> valueIteration(MDP& m, bool maximise=true); // for finite horizon
    std::unordered_map<State, float, StateHash> sspGetVaR(
        State initState,
        std::shared_ptr<MDP> m,
        bool maximise,
        int numVarInterpPts
    );

    std::vector<State> getReachableStates(
            State initState,
            std::shared_ptr<MDP> pMDP,
            bool maximise
    );

    std::tuple<std::string, std::unordered_map<State, float, StateHash>> getOptimalAction(
        std::shared_ptr<MDP> pMDP,
        State currentState,
        float currentAlpha,
        GRBEnv env,
        bool maximise=true
    );

    CvarHist executeEpisode(
        std::shared_ptr<MDP> pExpectedMDP,
        std::shared_ptr<MDP> pTrueMDP,
        State initialState,
        float initialAlpha
    );

    CvarHist sspExecuteEpisode(
            std::shared_ptr<MDP> pMDP,
            State initialState,
            float initialAlpha,
            std::unordered_map<State, std::string, StateHash> worstCasePolicy,
            bool maximise,
            bool perturbedProbs=false
    );

    CvarHist sspExecuteEpisodeLexicographic(
            std::shared_ptr<MDP> pMDP,
            State initialState,
            float initialAlpha,
            std::unordered_map<State, std::string, StateHash> worstCasePolicy,
            CvarLexicographic lexSolver,
            bool maximise
    );

    CvarHist executeBamdpEpisode(
            std::shared_ptr<MDP> pBamdp,
            std::shared_ptr<MDP> pTrueMDP,
            State initialState,
            float initialAlpha
    );

    float interpVarAlphaValue(
            std::unordered_map<State, float, StateHash>& valueFunction,
            State s,
            float alpha
    );

    std::unordered_map<State, std::map<std::string, std::string>, StateHash> getCvarPolicy(){
        return cvarPolicy;
    }

    std::unordered_map<State, std::map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> getCvarPerturbationPolicy(){
        return cvarPerturbationPolicy;
    }

};

#endif
