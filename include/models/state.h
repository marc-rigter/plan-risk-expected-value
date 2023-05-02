#ifndef state_factor
#define state_factor
#include <string>
#include <iostream>
#include <unordered_map>
#include <boost/functional/hash.hpp>

/* State class to represent a class in an MDP.

Attributes:
    stateMapping: a mapping from string state factors to string state factor
        values.
*/
class State
{
private:
    std::unordered_map<std::string, std::string> stateMapping;
    std::map<std::string, std::string> orderedStateMapping;

public:
    State();
    State(std::unordered_map<std::string, std::string> mapping);
    bool operator==(const State sf) const;
    bool operator!=(const State sf) const;
    std::string getValue(std::string sf) const;
    std::unordered_map<std::string, std::string> getStateMapping() const;
    std::map<std::string, std::string> getOrderedStateMapping() const;
    std::vector<std::string> getStateFactors() const;
    std::string toString() const;

    friend std::ostream& operator<< (std::ostream &out, const State &s){
        for(auto pair : s.getOrderedStateMapping()){
            out << pair.first << ": " << pair.second << ", ";
        }
        return out;
    }
};

struct StateHash
{
  std::size_t operator()(const State& stateToHash) const
  {
      using boost::hash_value;
      using boost::hash_combine;

      // Start with a hash value of 0    .
      std::size_t seed = 0;

      std::map<std::string, std::string> stateMap = stateToHash.getOrderedStateMapping();
      for(auto const &pair : stateMap){
          hash_combine(seed, pair.second);
      }

      // Return the result.
      return seed;
  }
};


#endif
