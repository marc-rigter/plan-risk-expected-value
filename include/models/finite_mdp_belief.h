#ifndef finite_mdp_belief
#define finite_mdp_belief
#include <string>
#include <unordered_map>
#include "mdp.h"
#include "belief.h"


/* Class for a belief state defined by a finite set of MDPs each weighted by
a probability.
*/
class FiniteMDPBelief : public Belief
{
private:
    std::unordered_map<std::shared_ptr<MDP>, float> weights;

public:
    FiniteMDPBelief(std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights_);
    void updateBelief(State initState, std::string action, State nextState);
    std::unordered_map<State, float, StateHash> getBeliefTransitionProbs(State state, std::string action) const;
    std::shared_ptr<MDP> sampleModel();
    std::unordered_map<std::shared_ptr<MDP>, float> getWeights() const;
    std::shared_ptr<MDP> getExpectedMDP() const;
    std::unordered_map<std::shared_ptr<MDP>, float> getParticleWeights(int numParticles);

    FiniteMDPBelief* clone () const        // Virtual constructor (copying)
    {
      return new FiniteMDPBelief (*this);
    }

    // copy constructor
    FiniteMDPBelief(const FiniteMDPBelief &b2) {
        weights = b2.weights;
    }

    int getMaxNumSamples() const{
        return weights.size();
    }
};

#endif
