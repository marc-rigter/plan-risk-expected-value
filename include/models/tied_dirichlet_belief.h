#ifndef tied_dirichlet_belief
#define tied_dirichlet_belief
#include <string>
#include <unordered_map>
#include <bits/stdc++.h>
#include "mdp.h"
#include "belief.h"
#include "tied_dirichlet_distribution.h"

/* Class for a belief where there is a separate dirichlet for each action, but
these are tied for different states.
*/
class TiedDirichletBelief : public Belief
{
public:
    typedef std::unordered_map<std::string, State> successorMapping;

    TiedDirichletBelief(
        std::shared_ptr<MDP> pMDP_,
        std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> dirichletDistributions_,
        std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping_
    );

    void updateBelief(State initState, std::string action, State nextState);
    std::shared_ptr<MDP> getExpectedMDP() const;
    std::shared_ptr<MDP> sampleModel();

    std::unordered_map<State, float, StateHash> getBeliefTransitionProbs(
            State state,
            std::string action) const;

    std::unordered_map<State, float, StateHash> getSampleTransitionProbs(
            State state,
            std::string action,
            std::unordered_map<std::shared_ptr<TiedDirichletDistribution>, std::unordered_map<std::string, float>> dists);

    TiedDirichletBelief* clone () const        // Virtual constructor (copying)
    {
      return new TiedDirichletBelief (*this);
    }

    // copy constructor
    TiedDirichletBelief(const TiedDirichletBelief &b2) {

        // mapping from old pointers to action and new pointer
        std::unordered_map<std::shared_ptr<TiedDirichletDistribution>, std::shared_ptr<TiedDirichletDistribution>> dirichletMap;
        for(auto kv : b2.dirichletDistributions){

            // if pointer is not in map
            if(dirichletMap.find(kv.second) == dirichletMap.end()){
                std::shared_ptr<TiedDirichletDistribution> newDirichlet(kv.second->clone());
                dirichletMap[kv.second] = newDirichlet;
            }
        }

        std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> newDirichlets;
        for(auto kv : b2.dirichletDistributions){
            newDirichlets[kv.first] = dirichletMap[kv.second];
        }

        dirichletDistributions = newDirichlets;
        pseudoStateMapping = b2.pseudoStateMapping;
    }

    int getMaxNumSamples() const{
        return INT_MAX;
    }
    std::unordered_map<std::shared_ptr<MDP>, float> getParticleWeights(int numParticles);

private:
    std::shared_ptr<MDP> pMDP;
    std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> dirichletDistributions;
    std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping;

};
#endif
