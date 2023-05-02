#ifndef hist
#define hist
#include <string>
#include <iostream>
#include "state.h"

std::string getNewHistory(State prevHistoryState, std::string action, State nextState);

class Hist
{
private:
    std::vector<std::tuple<State, std::string, float>> path;

protected:
    float totalReturn;


public:
    Hist();
    void addTransition(State s, std::string action, float reward);
    virtual void printPath();
    float getTotalReturn();
};

#endif
