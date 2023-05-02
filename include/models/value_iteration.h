#ifndef value_iteration
#define value_iteration
#include <string>
#include "state.h"
#include "mdp.h"
#include <unordered_map>


class VI
{
public:
    VI(){};

    std::tuple<std::unordered_map<State, float, StateHash>, std::unordered_map<State, std::string, StateHash>> valueIteration(MDP& mdp, bool maxReward);
    std::tuple<float, std::string> bellmanUpdate(
                                        MDP& m,
                                        State s,
                                        bool maxReward,
                                        std::unordered_map<State, float, StateHash>& value
                                    );

    virtual float getQVal(
        MDP& m,
        State s,
        std::string action,
        std::unordered_map<State, float, StateHash> value,
        bool maximise
    );
};


#endif
