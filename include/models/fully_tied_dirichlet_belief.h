#ifndef fully_tied_dirichlet_belief
#define fully_tied_dirichlet_belief
#include <string>
#include <unordered_map>
#include <bits/stdc++.h>
#include "mdp.h"
#include "belief.h"
#include "tied_dirichlet_distribution.h"

/* Class for a belief where there is a single dirichlet distribution that governs
the successor probabilities for all state-action pairs.
*/
class FullyTiedDirichletBelief : public Belief
{
public:
    typedef std::unordered_map<std::string, State> successorMapping;

    FullyTiedDirichletBelief(
        std::shared_ptr<MDP> pMDP_,
        TiedDirichletDistribution dirichletDistribution_,
        std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping_
    );

    void updateBelief(State initState, std::string action, State nextState);
    std::shared_ptr<MDP> getExpectedMDP() const;
    TiedDirichletDistribution getDirichletDistribution() const;
    std::shared_ptr<MDP> sampleModel();

    std::unordered_map<State, float, StateHash> getBeliefTransitionProbs(
            State state,
            std::string action) const;

    std::unordered_map<State, float, StateHash> getSampleTransitionProbs(
            State state,
            std::string action,
            std::unordered_map<std::string, float> dist) const;

    FullyTiedDirichletBelief* clone () const        // Virtual constructor (copying)
    {
      return new FullyTiedDirichletBelief (*this);
    }

    // copy constructor
    FullyTiedDirichletBelief(const FullyTiedDirichletBelief &b2) {
        dirichletDistribution = b2.dirichletDistribution;
        pseudoStateMapping = b2.pseudoStateMapping;
    }

    int getMaxNumSamples() const{
        return INT_MAX;
    }
    std::unordered_map<std::shared_ptr<MDP>, float> getParticleWeights(int numParticles);

private:
    std::shared_ptr<MDP> pMDP;
    TiedDirichletDistribution dirichletDistribution;
    std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping;

};
#endif
