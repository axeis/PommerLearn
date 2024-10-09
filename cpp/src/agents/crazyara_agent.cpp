#include "crazyara_agent.h"
#include <limits>

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"

PlanningAgentType planning_agent_type_from_string(std::string str)
{
    if (str == "None")
    {
        return PlanningAgentType::None;
    }
    else if (str == "SimpleUnbiasedAgent")
    {
        return PlanningAgentType::SimpleUnbiasedAgent;
    }
    else if (str == "SimpleAgent")
    {
        return PlanningAgentType::SimpleAgent;
    }
    else if (str == "LazyAgent")
    {
        return PlanningAgentType::LazyAgent;
    }
    else if (str == "RawNetAgent" || str == "RawNetworkAgent")
    {
        return PlanningAgentType::RawNetworkAgent;
    }

    throw std::runtime_error(std::string("Unknown planning agent type: ") + str);
}

void CrazyAraAgent::init_state(bboard::GameMode gameMode, bboard::ObservationParameters obsParams, bboard::ObservationParameters opponentObsParams, bool useVirtualStep, bool trackStats)
{
    pommermanState = std::make_unique<PommermanState>(gameMode, has_stateful_model(), 800);
    pommermanState->set_agent_observation_params(obsParams);
    pommermanState->set_opponent_observation_params(opponentObsParams);
    pommermanState->set_virtual_step(useVirtualStep);
    pommermanState->set_track_stats(trackStats);
}

void CrazyAraAgent::use_environment_state(bboard::Environment* env) 
{
    this->env = env;
}

void _get_policy(EvalInfo* info, float* policyProbs) {
    std::fill_n(policyProbs, NUM_MOVES, 0);
    for (size_t i = 0; i < info->legalMoves.size(); i++) {
        Action a = info->legalMoves.at(i);
        float prob = info->policyProbSmall.at(i);

        size_t index = StateConstantsPommerman::action_to_index(a);
        policyProbs[index] = prob;
    }
}

void _get_q(EvalInfo* info, float* q) {
    std::fill_n(q, NUM_MOVES, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < info->legalMoves.size(); i++) {
        Action a = info->legalMoves.at(i);
        float qVal = info->qValues.at(i);

        size_t index = StateConstantsPommerman::action_to_index(a);
        q[index] = qVal;
    }
}

void CrazyAraAgent::set_pommerman_state(const bboard::Observation *obs)
{
    if (env) {
        bboard::State& state = env->GetState();
        // sanity check: the given observation should originate from the
        // same environment and therefore have the same time step
        assert(state.timeStep == obs->timeStep);
        pommermanState->set_state(&state);
    }
    else {
        pommermanState->set_observation(obs);
    }
}

void CrazyAraAgent::print_pv_line(MCTSAgent* mctsAgent, const bboard::Observation *obs)
{
    if (mctsAgent != nullptr) {
        if (evalInfo.pv.size() > 0) {
            std::cout << "BEGIN ========== Principal Variation" << std::endl;
            set_pommerman_state(obs);
            std::cout << "(" << pommermanState->state.timeStep << "): Initial state" << std::endl;
            pommermanState->state.Print(false);
            for (Action a : evalInfo.pv[0]) {
                pommermanState->do_action(a);
                std::cout << "(" << pommermanState->state.timeStep << "): after " << StateConstantsPommerman::action_to_uci(a, false) << std::endl;
                pommermanState->state.Print(false);
            }
            std::cout << "END ========== " << std::endl;
        }
    }
}

const PommermanState* CrazyAraAgent::get_pommerman_state() const
{
    return pommermanState.get();
}

void CrazyAraAgent::add_results_to_buffer(const NeuralNetAPI* net, bboard::Move bestAction)
{
    if (has_buffer()) {
        if (!planeBuffer) {
            // create buffers on the fly
            planeBuffer = std::unique_ptr<float[]>(new float[PLANES_TOTAL_FLOATS]);
            policyBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
            qBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
        }
        pommermanState->get_state_planes(true, planeBuffer.get(), net->get_version());
        _get_policy(&evalInfo, policyBuffer.get());
        _get_q(&evalInfo, qBuffer.get());

        sampleBuffer->addSample(planeBuffer.get(), bestAction, policyBuffer.get(), evalInfo.bestMoveQ.at(0), qBuffer.get());
    }
}

bboard::Move CrazyAraAgent::act(const bboard::Observation *obs)
{
    set_pommerman_state(obs);
    crazyara::Agent* agent = get_acting_agent();
    NeuralNetAPI* net = get_acting_net();
    agent->set_search_settings(pommermanState.get(), &searchLimits, &evalInfo);
    agent->perform_action();

    bboard::Move bestAction = bboard::Move(agent->get_best_action());

    #if ! defined(DISABLE_UCI_INFO) || defined(CRAZYARA_AGENT_PV)
        MCTSAgent* mctsAgent = dynamic_cast<MCTSAgent*>(agent);
    #endif

    #ifndef DISABLE_UCI_INFO
        if (mctsAgent != nullptr) {
            mctsAgent->print_root_node();
        }
    #endif

    #ifdef CRAZYARA_AGENT_PV
        print_pv_line(mctsAgent, obs);
    #endif

    add_results_to_buffer(net, bestAction);
    eval_time_ms.update(std::chrono::duration_cast<std::chrono::milliseconds>(evalInfo.end - evalInfo.start).count());
    eval_depth.update(evalInfo.depth);
    eval_depth_sel.update(evalInfo.selDepth);
    eval_nodes.update(evalInfo.nodes);
    return bestAction;
}

void CrazyAraAgent::reset() {
    // update ID of the agent of MCTS states
    pommermanState->set_agent_id(id);
    // currently stored state is not valid anymore
    pommermanState->hasValidState = false;
}

void CrazyAraAgent::fill_single_nn_vector(const string& modelDirectory, vector<unique_ptr<NeuralNetAPI>>& netSingleVector, 
                                            vector<vector<unique_ptr<NeuralNetAPI>>>& netBatchesVector, const SearchSettings& searchSettings, const int deviceID)
{
    unique_ptr<NeuralNetAPI> netSingleTmp = load_network(modelDirectory, deviceID, 1);
    netSingleTmp->validate_neural_network();
    netSingleVector.push_back(std::move(netSingleTmp));

    size_t idx = 0;
    for (int deviceId = deviceID; deviceId <= deviceID; ++deviceId) {
        for (size_t i = 0; i < size_t(searchSettings.threads); ++i) {
            unique_ptr<NeuralNetAPI> netBatchesTmp = load_network(modelDirectory, deviceId, searchSettings.batchSize);
            netBatchesTmp->validate_neural_network();
            netBatchesVector[idx].push_back(std::move(netBatchesTmp));
            ++idx;
        }
    }
}

void CrazyAraAgent::fill_nn_vectors(const string& modelDirectory, vector<unique_ptr<NeuralNetAPI>>& netSingleVector, vector<vector<unique_ptr<NeuralNetAPI>>>& netBatchesVector, const SearchSettings& searchSettings, const int deviceID)
{
    netSingleVector.clear();
    netBatchesVector.clear();
    // threads is the first dimension, the phase are the 2nd dimension
    netBatchesVector.resize(searchSettings.threads * 1);

    // early return if no phases are used
    for (const auto& entry : fs::directory_iterator(modelDirectory)) {
        if (!fs::is_directory(entry.path())) {
            fill_single_nn_vector(modelDirectory, netSingleVector, netBatchesVector, searchSettings, deviceID);
            return;
        }
        else {
            break;
        }
    }

    // analyse directory to get num phases
    for (const auto& entry : fs::directory_iterator(modelDirectory)) {
        std::cout << entry.path().generic_string() << std::endl;

        fill_single_nn_vector(entry.path().generic_string(), netSingleVector, netBatchesVector, searchSettings, deviceID);
    }
}

std::unique_ptr<NeuralNetAPI> CrazyAraAgent::load_network(const std::string& modelDirectory, const int deviceID, unsigned int batchSize)
{
#ifdef TENSORRT
    return std::make_unique<TensorrtAPI>(deviceID, batchSize, modelDirectory, "float32");
#elif defined (TORCH)
    return std::make_unique<TorchAPI>("cpu", 0, 1, modelDirectory);
#endif
}
