#include "crazyara_agent.h"
#include "depth_switch_agent.h"

MCTSCrazyAraAgent::MCTSCrazyAraAgent(const std::string& modelDirectory, const int deviceID, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits):
    modelDirectory(modelDirectory), deviceID(deviceID)
{
    this->playSettings = playSettings;
    this->searchSettings = searchSettings;
    this->searchLimits = searchLimits;
    this->netSingleVector.clear();
    vector<unique_ptr<NeuralNetAPI>> tempNetVector;
    
    for (const auto& entry : fs::directory_iterator(modelDirectory)) {
        unique_ptr<NeuralNetAPI> netSingleTemp = load_network(entry.path().generic_string(), deviceID);
        netSingleTemp->validate_neural_network();
        vector<unique_ptr<NeuralNetAPI>> netBatchesTemp = load_network_batches(entry.path().generic_string(), deviceID, searchSettings);
        netBatchesTemp.front()->validate_neural_network();
       

        this->netSingleVector.push_back(std::move(netSingleTemp));
        this->netBatchesVector.push_back(std::move(netBatchesTemp));

        /* 
        later on this wrapper needs to access the networks. CrazyAra currently does not allow this.
        this ist a workaround to save a copy in this scope while CrazyAra takes Ownership of its own
        */ 
        netSingleTemp = load_network(entry.path().generic_string(), deviceID);
        tempNetVector.push_back(std::move(netSingleTemp));
        

    }
    // check for stateful because netSingleVector is empty after next call
    this->modelIsStateful = netSingleVector.at(0)->has_auxiliary_outputs();
    agent = std::make_unique<MCTSAgent>(tempNetVector, this->netBatchesVector, &this->searchSettings, &this->playSettings);
}

vector<unique_ptr<NeuralNetAPI>> MCTSCrazyAraAgent::load_network_batches(const string& modelDirectory, const int deviceID, const SearchSettings& searchSettings)
{
    vector<unique_ptr<NeuralNetAPI>> netBatches;
#ifdef MXNET
    #ifdef TENSORRT
        const bool useTensorRT = bool(Options["Use_TensorRT"]);
    #else
        const bool useTensorRT = false;
    #endif
#endif
    int First_Device_ID = deviceID;
    int Last_Device_ID = deviceID;
    for (int deviceId = First_Device_ID; deviceId <= Last_Device_ID; ++deviceId) {
        for (size_t i = 0; i < searchSettings.threads; ++i) {
    #ifdef MXNET
            netBatches.push_back(make_unique<MXNetAPI>(Options["Context"], deviceId, searchSettings.batchSize, modelDirectory, useTensorRT));
    #elif defined TENSORRT
            netBatches.push_back(make_unique<TensorrtAPI>(deviceId, searchSettings.batchSize, modelDirectory, "float16"));
    #elif defined TORCH
            netBatches.push_back(make_unique<TorchAPI>("cpu", deviceId, searchSettings.batchSize, modelDirectory));
    #endif
        }
    }
    netBatches[0]->validate_neural_network();
    return netBatches;
}

SearchSettings MCTSCrazyAraAgent::get_default_search_settings(const bool selfPlay)
{
    SearchSettings searchSettings;

    searchSettings.batchSize = 8;
    searchSettings.threads = 4;
    searchSettings.useMCGS = false;
    searchSettings.multiPV = 1;
    if (selfPlay)
    {
        searchSettings.dirichletEpsilon = 0.25f;
        searchSettings.nodePolicyTemperature = 1.0f;
    }
    else
    {
        searchSettings.dirichletEpsilon = 0;
        searchSettings.nodePolicyTemperature = 1.7f;
    }
    searchSettings.reuseTree = false;
    searchSettings.mctsSolver = false;
    searchSettings.searchPlayerMode = MODE_SINGLE_PLAYER;
    return searchSettings;
}

void MCTSCrazyAraAgent::set_planning_agents(PlanningAgentType planningAgentType, PlanningAgentType planningAgentTeamType, int switchDepth)
{
    this->planningAgentType = planningAgentType;
    this->planningAgentTeamType = planningAgentTeamType;
    this->switchDepth = switchDepth;
}

void MCTSCrazyAraAgent::update_planning_agents()
{
    if (!pommermanState) {
        throw std::runtime_error("The state has to be created before planning agents can be assigned to it.");
    }
    if (pommermanState->agentID == -1) {
        // state is not initialized yet, e.g. when we clone an agent before it gets its first reset.
        return;
    }

    int ownTeam = bboard::GetTeam(pommermanState->gameMode, pommermanState->agentID); 

    // create raw net agent queue that is shared by all planning raw net agents
    std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue;
    if (planningAgentType == PlanningAgentType::RawNetworkAgent || (ownTeam != 0 && planningAgentTeamType == PlanningAgentType::RawNetworkAgent)) {
        if (modelDirectory.empty()) {
            throw std::runtime_error("Cannot use planningAgentType RawNetworkAgent with empty model directory.");
        }

        rawNetAgentQueue = RawCrazyAraAgent::load_raw_net_agent_queue(modelDirectory, searchSettings.threads, deviceID);
    }

    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        // we do not need a planning agent for ourselves
        if (pommermanState->agentID == i) {
            continue;
        }

        // temporary variable, will be moved to pommermanState
        std::unique_ptr<Clonable<bboard::Agent>> agent;

        // get correct planning agent type
        PlanningAgentType agentType;
        if (ownTeam != 0 && ownTeam == bboard::GetTeam(pommermanState->gameMode, i)) {
            agentType = this->planningAgentTeamType;
        }
        else {
            agentType = this->planningAgentType;
        }

#ifndef DISABLE_UCI_INFO
        std::cout << "Agent " << pommermanState->agentID << " > planning agent " << i << " with type " << agentType << std::endl;
#endif

        // create the agent
        switch (agentType)
        {
        case PlanningAgentType::None:
        {
            continue;            
        }
        case PlanningAgentType::SimpleUnbiasedAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::SimpleUnbiasedAgent>(agents::SimpleUnbiasedAgent(rand()))
            );
            break;
        }
        case PlanningAgentType::SimpleAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand()))
            );
            break;
        }
        case PlanningAgentType::LazyAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::LazyAgent>(agents::LazyAgent())
            );
            break;
        }
        case PlanningAgentType::RawNetworkAgent:
        {
            std::unique_ptr<RawCrazyAraAgent> crazyAraAgent = std::make_unique<RawCrazyAraAgent>(rawNetAgentQueue);
            crazyAraAgent->id = i;
            crazyAraAgent->init_state(pommermanState->gameMode, pommermanState->opponentObsParams, pommermanState->opponentObsParams, pommermanState->useVirtualStep, pommermanState->trackStats);
            agent = std::move(crazyAraAgent);
            break;
        }
        default:
            break;
        }

        if (agentType == PlanningAgentType::RawNetworkAgent && switchDepth >= 0) {
            agent = std::make_unique<DepthSwitchAgent>(std::move(agent), switchDepth, rand());
        }

        pommermanState->set_planning_agent(std::move(agent), i);
    }
}

bool MCTSCrazyAraAgent::has_stateful_model()
{
    return this->modelIsStateful;
}

crazyara::Agent* MCTSCrazyAraAgent::get_acting_agent()
{
    return agent.get();
}

NeuralNetAPI* MCTSCrazyAraAgent::get_acting_net()
{
    //TODO will not perform MOE 


    return netSingleVector.at(0).get(); 
}

void MCTSCrazyAraAgent::reset()
{
    CrazyAraAgent::reset();
    // initialize / update planning agents
    if (previousAgentID != pommermanState->agentID) {
        update_planning_agents();
        previousAgentID = pommermanState->agentID;
    }
}

bboard::Agent* MCTSCrazyAraAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> MCTSCrazyAraAgent::clone()
{
    if (!pommermanState.get()) {
        throw std::runtime_error("Cannot clone agent with uninitialized state!");
    }

    std::unique_ptr<MCTSCrazyAraAgent> clonedAgent = std::make_unique<MCTSCrazyAraAgent>(modelDirectory, deviceID, playSettings, searchSettings, searchLimits);

    clonedAgent->id = id;
    clonedAgent->previousAgentID = previousAgentID;
    clonedAgent->pommermanState = std::unique_ptr<PommermanState>(pommermanState->clone());
    // we have to reset the planning agents to disconnect the clones if they use shared networks
    clonedAgent->set_planning_agents(planningAgentType, planningAgentTeamType, switchDepth);
    clonedAgent->update_planning_agents();

    return clonedAgent;
}
