#ifndef cvar_lexicographic
#define cvar_lexicographic
#include <string>
#include "state.h"
#include "mdp.h"
#include "gurobi_c++.h"
#include "cvar_hist.h"
#include "cvar_value_iteration.h"

class CvarLexicographic
{
private:
    std::vector<std::string> costInterpPts;
    std::vector<float> costInterpFloats;
    float VaR;
    std::unordered_map<State, float, StateHash> worstCaseValue;
    std::unordered_map<State, std::string, StateHash> worstCasePolicy;
    std::unordered_map<State, float, StateHash> value;

public:
    CvarLexicographic(
        int numInterpPts_,
        float VaR_,
        std::unordered_map<State, float, StateHash> worstCaseValue_,
        std::unordered_map<State, std::string, StateHash> worstCasePolicy_
    );

    std::unordered_map<State, std::vector<std::string>, StateHash> getPrunedActions(
        std::shared_ptr<MDP> pMDP
    );

    std::unordered_map<State, float, StateHash> computeLexicographicValue(
        std::shared_ptr<MDP> pMDP,
        bool isSSP
    );

    float getQValue(
        std::shared_ptr<MDP> pMDP,
        State baseState,
        float costSoFar,
        std::string action
    );

    std::unordered_map<State, float, StateHash> approximateVI(
        std::shared_ptr<MDP> pMDP,
        std::unordered_map<State, std::vector<std::string>, StateHash> prunedActions,
        bool isSSP = true
    );

    std::string getOptimalAction(
        std::shared_ptr<MDP> pMDP,
        State baseState,
        float costSoFar
    );

    void approximateVIBackup(
        State augmentedState,
        std::shared_ptr<MDP> pMDP,
        std::unordered_map<State, float, StateHash>& val,
        std::unordered_map<State, std::vector<std::string>, StateHash>& prunedActions
    );

    std::vector<std::string> getAllowedActions(
        std::shared_ptr<MDP> pMDP,
        State baseState,
        float costSoFar
    );

    float interpCostValue(
        std::unordered_map<State, float, StateHash>& value,
        State augmentedState
    );
};

#endif
