#include <iostream>
#include <cmath>
#include <unordered_map>
#include "state.h"

State::State(std::unordered_map<std::string, std::string> mapping){
    stateMapping = mapping;
    std::map<std::string, std::string> ordered(mapping.begin(), mapping.end());
    orderedStateMapping = ordered;
}

State::State(){

}

std::string State::getValue(std::string sf) const{
    return this->stateMapping.at(sf);
}

std::unordered_map<std::string, std::string> State::getStateMapping() const{
    return this->stateMapping;
}

std::map<std::string, std::string> State::getOrderedStateMapping() const{
    return this->orderedStateMapping;
}

std::string State::toString() const{
    std::string mystr;
    for(auto kv : this->orderedStateMapping){
        mystr.append(kv.first).append(":").append(kv.second).append(";");
    }
    return mystr;
}

std::vector<std::string> State::getStateFactors() const{
    std::vector<std::string> sf;
    for(auto kv : this->stateMapping){
        sf.push_back(kv.first);
    }
    return sf;
}

/* override the equality operator by checking for equality of each of the
state factors.

Returns:
    boolean defining whether the state to be compared is equal.
*/
bool State::operator==(const State sf) const
{
    bool equal = true;
    for(auto const &pair: this->stateMapping){
        if(this->stateMapping.at(pair.first) != sf.getStateMapping().at(pair.first)){
            equal = false;
        }
    }

    return equal;
}

bool State::operator!=(const State sf) const
{
    return !(*this == sf);
}
