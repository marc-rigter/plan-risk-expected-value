#ifndef worst_case_value_iteration
#define worst_case_value_iteration
#include <string>
#include "state.h"
#include "mdp.h"
#include <unordered_map>
#include "value_iteration.h"


class worstCaseVI : public VI
{
public:
    worstCaseVI(){};

    float getQVal(
        MDP& m,
        State s,
        std::string action,
        std::unordered_map<State, float, StateHash> value,
        bool maximise
    );
};


#endif
