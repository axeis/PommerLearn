#include "pymethods.hpp"
#include "agents.hpp"
#include "agents/crazyara_agent.h"

#include <iostream>
#include "string.h"

std::unique_ptr<CrazyAraAgent> create_crazyara_agent(std::string modelDir, int deviceID, uint stateSize, bool rawNetAgent, bool ffa, bool virtualStep, SearchLimits searchLimits=SearchLimits())
{
    StateConstants::init(false);
    StateConstantsPommerman::set_auxiliary_outputs(stateSize);
    
    std::unique_ptr<CrazyAraAgent> crazyAraAgent;
    if(rawNetAgent)
    {
        crazyAraAgent = std::make_unique<RawCrazyAraAgent>(modelDir, deviceID);
    }
    else
    {
        SearchSettings searchSettings = MCTSCrazyAraAgent::get_default_search_settings(false);
        PlaySettings playSettings;
        crazyAraAgent = std::make_unique<MCTSCrazyAraAgent>(modelDir, deviceID, playSettings, searchSettings, searchLimits);
    }

    // partial observability
    bboard::ObservationParameters obsParams;
    if (ffa){
        obsParams.exposePowerUps = false;
    }
    else {
        obsParams.exposePowerUps = false;
    }
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::OnlySelf;
    obsParams.exposePowerUps = false;


    if (ffa){
        crazyAraAgent->init_state(bboard::GameMode::FreeForAll, obsParams, obsParams, virtualStep);
    } 
    else {
        crazyAraAgent->init_state(bboard::GameMode::TwoTeams, obsParams, obsParams, virtualStep);
    }

    if(!rawNetAgent)
    {
        ((MCTSCrazyAraAgent*)crazyAraAgent.get())->init_planning_agents(PlanningAgentType::SimpleAgent);
    }

    return crazyAraAgent;
}

std::pair<std::string, std::string> _extract_arg(std::string input, std::string delimiter=":")
{
    auto delimiterPos = input.find(":");
    if(delimiterPos == std::string::npos)
    {
        return std::pair<std::string, std::string>(input, "");
    }

    std::string arg = input.substr(0, delimiterPos);
    std::string remainder = input.substr(delimiterPos + 1, input.length() - delimiterPos - 1);
    return std::pair<std::string, std::string>(arg, remainder);
}

std::unique_ptr<bboard::Agent> PyInterface::new_agent(std::string agentName, long seed)
{
    if(agentName == "SimpleAgent")
    {
        return std::make_unique<agents::SimpleAgent>(seed);
    }
    else if(agentName == "SimpleUnbiasedAgent")
    {
        return std::make_unique<agents::SimpleUnbiasedAgent>(seed);
    }
    else if(agentName.find("RawNetAgent") == 0)
    {
        auto tmp = _extract_arg(agentName);
        tmp = _extract_arg(tmp.second);
        std::string modelDir = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string stateSize = tmp.first;
        bool virtualStep = (tmp.second == "virtualStep");
        if(modelDir.empty() || stateSize.empty()) 
        {
            std::cout << "Could not parse agent identifier. Specify the agent like RawNet:dir:stateSize" << std::endl;
            return nullptr;
        }

        bool isFFA = !(agentName.find("RawNetAgentTeam") == 0);

        std::cout << "Creating RawNetAgent with " << std::endl
            << "> Model dir: " << modelDir << std::endl
            << "> State size: " << stateSize << std::endl
            << "> FFA: " << isFFA << std::endl
            << "> VirtualStep: " << virtualStep << std::endl;

        // always use device with id 0
        int deviceID = 0;
        return create_crazyara_agent(modelDir, deviceID, std::stoi(stateSize), true, isFFA, virtualStep);
    }
    else if(agentName.find("CrazyAraAgent") == 0)
    {
        auto tmp = _extract_arg(agentName);
        tmp = _extract_arg(tmp.second);
        std::string modelDir = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string stateSize = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string simulations = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string moveTime = tmp.first;
        bool virtualStep = (tmp.second == "virtualStep");
        if(modelDir.empty() || stateSize.empty() || simulations.empty() || moveTime.empty()) 
        {
            std::cout << "Could not parse agent identifier. Specify the agent like CrazyAra:dir:stateSize:simulations:moveTime (:virtualStep (optional)" << std::endl;
            return nullptr;
        }
        
        bool isFFA = !(agentName.find("CrazyAraAgentTeam") == 0);

        std::cout << "Creating CrazyAraAgent with " << std::endl
                  << "> Model dir: " << modelDir << std::endl
                  << "> State size: " << stateSize << std::endl
                  << "> Simulations: " << simulations << std::endl
                  << "> Movetime: " << moveTime << std::endl
                  << "> FFA: " << isFFA << std::endl
                  << "> VirtualStep: " << virtualStep << std::endl;

        SearchLimits searchLimits;
        searchLimits.simulations = std::stoi(simulations);
        searchLimits.movetime = std::stoi(moveTime);

        
        // always use device with id 0
        int deviceID = 0;
        return create_crazyara_agent(modelDir, deviceID, std::stoi(stateSize), false, isFFA, virtualStep,searchLimits=searchLimits);
    }

    return nullptr;
}
