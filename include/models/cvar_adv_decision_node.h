#ifndef cvar_adv_decision_node
#define cvar_adv_decision_node
#include <string>
#include <unordered_map>
#include <memory>
#include "mdp.h"
#include "state.h"
#include "cvar_decision_node.h"
#include "cvar_chance_node.h"
#include "bamdp_rollout_policy.h"

class CvarChanceNode;
class CvarDecisionNode;

/* class to be used for decision nodes of the adversary in CVaR
MDP optimisation. */
class CvarAdvDecisionNode : public CvarDecisionNode {
public:
    CvarAdvDecisionNode(
            std::shared_ptr<MDP> pMDP_,
            const State state_,
            const float alpha_,
            const std::string action_);

    int expandRandomPerturbation();
    int expandPerturbationBayesOpt();
    int expandNoPerturbation();
    int getNoPerturbationAction();
    void enumerateVertexPerturbations();
    bool doNotExpand();
    std::unordered_map<State, float, StateHash> getNoPerturbationMapping();
    std::unordered_map<int, std::shared_ptr<CvarChanceNode>>  getChanceChildren();
    std::unordered_map<State, float, StateHash> getPerturbationMapping(int index);
    std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> sampleRandomPerturbations(int numSamples);
    std::unordered_map<State, float, StateHash> sampleMaxExpectedImprovementPerturbation();
    virtual std::unordered_map<State, float, StateHash> getSuccessorProbs();
    int selectPerturbationActionUCB(float biasFactor=3.0f);
    int getBestAdvAction();
    std::string getAction();
    virtual std::shared_ptr<CvarChanceNode> createChild(std::unordered_map<State, float, StateHash> perturbationMapping);
    int addChildChanceNode(std::unordered_map<State, float, StateHash> pert);
    void updatePerturbationBudget();

protected:
    const std::string action;
    std::unordered_map<int, std::unordered_map<State, float, StateHash>> actionMapping;
    std::unordered_map<int, std::shared_ptr<CvarChanceNode>> chanceNodeChildren;
    std::unordered_map<State, float, StateHash> advPerturbation;
    std::vector<std::unordered_map<State, float, StateHash>> vertexPerturbations;
    bool verticesEnumerated = false;
    bool perturbationsSampled = false;
    std::vector<std::unordered_map<State, float, StateHash>> sampledPerturbations;
    float perturbationBudget;
};

#endif
