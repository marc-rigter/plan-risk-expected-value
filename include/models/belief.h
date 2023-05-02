#ifndef belief
#define  belief
#include <string>
#include <memory>
#include <unordered_map>
#include "mdp.h"
#include "state.h"



/* Base class for belief states.
*/
class Belief
{
private:

public:
    virtual ~Belief () {}
    virtual Belief * clone () const = 0;
    virtual void updateBelief(State initState, std::string action, State nextState) = 0;
    virtual std::unordered_map<State, float, StateHash> getBeliefTransitionProbs(State state, std::string action) const = 0;
    virtual std::shared_ptr<MDP> getExpectedMDP() const = 0;
    virtual std::shared_ptr<MDP> sampleModel() = 0;
    virtual int getMaxNumSamples() const = 0;
    virtual std::unordered_map<std::shared_ptr<MDP>, float> getParticleWeights(int numParticles) = 0;

    State sampleSuccessor(
        State s,
        std::string action,
        std::unordered_map<State, float, StateHash> successorProbs = {}) const;

    State samplePerturbedSuccessor(
        State s,
        std::string action,
        std::unordered_map<State, float, StateHash> perturbation,
        std::unordered_map<State, float, StateHash> successorProbs = {}) const;

    std::shared_ptr<MDP> toBamdp(std::shared_ptr<MDP> pMDP, State initState, int horizon);
};

#endif
