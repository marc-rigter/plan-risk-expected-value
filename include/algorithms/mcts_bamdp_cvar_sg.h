#ifndef mcts_bamdp_cvar_sg
#define mcts_bamdp_cvar_sg
#include <string>
#include <memory>
#include "belief.h"
#include "state.h"
#include "mcts.h"
#include "bamdp_cvar_decision_node.h"
#include "bamdp_cvar_chance_node.h"
#include "mcts_cvar_sg.h"

/* Class to implement monte carlo tree search to solve game optimising CVaR
in Bayes-Adaptive MDPs
*/
class BamdpCvarMCTS : public CvarMCTS {
public:
    BamdpCvarMCTS(float biasMultiplier_=3.0, float wideningParam_=0.3, std::string wideningStrategy_="random") : CvarMCTS(biasMultiplier_, wideningParam_, wideningStrategy_) {};

private:

};

#endif
