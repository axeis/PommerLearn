#include "crazyara_agent.h"

MCTSCrazyAraAgent::MCTSCrazyAraAgent(std::unique_ptr<NeuralNetAPI> singleNet, std::vector<std::unique_ptr<NeuralNetAPI>> netBatches, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits)
{
    this->playSettings = playSettings;
    this->searchSettings = searchSettings;
    this->searchLimits = searchLimits;
    this->singleNet = std::move(singleNet);
    this->netBatches = std::move(netBatches);
    agent = std::make_unique<MCTSAgent>(this->singleNet.get(), this->netBatches, &this->searchSettings, &this->playSettings);
}

MCTSCrazyAraAgent::MCTSCrazyAraAgent(const std::string& modelDirectory, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits):
    MCTSCrazyAraAgent(load_network(modelDirectory), load_network_batches(modelDirectory, searchSettings), playSettings, searchSettings, searchLimits)
{
    this->modelDirectory = modelDirectory;
}

vector<unique_ptr<NeuralNetAPI>> MCTSCrazyAraAgent::load_network_batches(const string& modelDirectory, const SearchSettings& searchSettings)
{
    vector<unique_ptr<NeuralNetAPI>> netBatches;
#ifdef MXNET
    #ifdef TENSORRT
        const bool useTensorRT = bool(Options["Use_TensorRT"]);
    #else
        const bool useTensorRT = false;
    #endif
#endif
    int First_Device_ID = 0;
    int Last_Device_ID = 0;
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

    searchSettings.virtualLoss = 1;
    searchSettings.batchSize = 8;
    searchSettings.threads = 2;
    searchSettings.useMCGS = false;
    searchSettings.multiPV = 1;
    searchSettings.nodePolicyTemperature = 1.0f;
    if (selfPlay)
    {
        searchSettings.dirichletEpsilon = 0.25f;
    }
    else
    {
        searchSettings.dirichletEpsilon = 0;
    }
    searchSettings.dirichletAlpha = 0.2f;
    searchSettings.epsilonGreedyCounter = 0;
    searchSettings.epsilonChecksCounter = 0;
    searchSettings.qVetoDelta = 0.4;
    searchSettings.qValueWeight = 1.0f;
    searchSettings.reuseTree = false;
    searchSettings.mctsSolver = false;

    return searchSettings;
}

void MCTSCrazyAraAgent::init_state(bboard::GameMode gameMode, bboard::ObservationParameters obsParams, bboard::ObservationParameters opponentObsParams, uint8_t valueVersion, PlanningAgentType planningAgentType)
{
    CrazyAraAgent::init_state(gameMode, obsParams, opponentObsParams, valueVersion, planningAgentType);

    // create raw net agent queue that is shared by all planning raw net agents
    std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue;
    if (planningAgentType == PlanningAgentType::RawNetworkAgent) {
        if (modelDirectory.empty()) {
            throw std::runtime_error("Cannot use planningAgentType RawNetworkAgent with empty model directory.");
        }

        rawNetAgentQueue = RawCrazyAraAgent::load_raw_net_agent_queue(modelDirectory, searchSettings.threads);
    }

    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        std::unique_ptr<Clonable<bboard::Agent>> agent;

        // other agents used for planning
        switch (planningAgentType)
        {
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
            crazyAraAgent->init_state(gameMode, opponentObsParams, opponentObsParams, valueVersion);
            agent = std::move(crazyAraAgent);
            break;
        }
        default:
            break;
        }

        pommermanState->set_planning_agent(std::move(agent), i);
    }
}

bool MCTSCrazyAraAgent::has_stateful_model()
{
    return singleNet->has_auxiliary_outputs();
}

crazyara::Agent* MCTSCrazyAraAgent::get_acting_agent()
{
    return agent.get();
}

NeuralNetAPI* MCTSCrazyAraAgent::get_acting_net()
{
    return singleNet.get();
}
