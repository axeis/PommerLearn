/*
 * @file: pommermanstate.cpp
 * Created on 15.07.2020
 * @author: queensgambit
 */

#include "pommermanstate.h"
#include "data_representation.h"
#include "agents.hpp"

#include <mutex>

uint StateConstantsPommerman::auxiliaryStateSize = 0;

PommermanState::PommermanState(bboard::GameMode gameMode, bool statefulModel, uint maxTimeStep):
    hasTrueState(false),
    agentID(-1),
    gameMode(gameMode),
    eventHash(0),
    statefulModel(statefulModel),
    maxTimeStep(maxTimeStep),
    hasPlanningAgents(false),
    hasBufferedActions(false),
    hasValidState(false)
{
#ifndef MCTS_SINGLE_PLAYER
    simulatedOpponentID = -1;
    myTurn = true;
#endif

    std::fill_n(moves, bboard::AGENT_COUNT, bboard::Move::IDLE);
    //std::fill_n(&this->vsParams.itemAge[0][0], bboard::BOARD_SIZE * bboard::BOARD_SIZE , 0);
    if (StateConstantsPommerman::NB_AUXILIARY_OUTPUTS() != 0) {
        auxiliaryOutputs.resize(StateConstantsPommerman::NB_AUXILIARY_OUTPUTS());
    }
    else if (statefulModel)
    {
        throw std::runtime_error("You have not set an auxiliary (state) output but you claim that your model is stateful.");
    }
}

#ifndef MCTS_SINGLE_PLAYER
int _get_closest_opponent(const bboard::State* state, const int id)
{
    const bboard::AgentInfo& self = state->agents[id];
    int closestOpponent = -1;
    int closestDistance = bboard::BOARD_SIZE * 2;
    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        if (i == id) {
            continue;
        }
        const bboard::AgentInfo& other = state->agents[i];
        if (other.dead || !self.IsEnemy(other)) {
            continue;
        }

        int manhattan = abs(self.x - other.x) + abs(self.y - other.y);
        if (manhattan < closestDistance) {
            closestDistance = manhattan;
            closestOpponent = i;
        }
    }

    return closestOpponent;
}
#endif

void PommermanState::set_agent_id(const int id)
{
    this->agentID = id;
}

void _init_search(PommermanState& pommermanState)
{
    if (pommermanState.hasPlanningAgents) {
        pommermanState.planning_agents_reset();
    }

#ifndef MCTS_SINGLE_PLAYER
    pommermanState.myTurn = true;
    pommermanState.simulatedOpponentID = _get_closest_opponent(&pommermanState.state, pommermanState.agentID);
#ifndef DISABLE_UCI_INFO
    std::cout << "Simulated opponent: " << pommermanState.simulatedOpponentID << std::endl;
#endif
#endif
}

void PommermanState::set_virtual_step(bool useVirtualStep)
{
    this->useVirtualStep = useVirtualStep;
}

void PommermanState::set_track_stats(bool trackStats)
{
    this->trackStats = trackStats;
}

void PommermanState::set_state(const bboard::State* state)
{
    // copy the state
    this->state = *state;
    this->hasTrueState = true;
    this->hasValidState = true;

    _init_search(*this);
}

void PommermanState::set_observation(const bboard::Observation* obs)
{
    auto obsCopy = *obs;

    if (this->hasValidState)
    {
        // check that the observation belongs to the next time step
        this->hasValidState = obsCopy.timeStep == state.timeStep + 1;
    }

    // initialize state
    if (obsCopy.timeStep == 0 || !this->hasValidState) {
        state = bboard::State();

        if (!this->trackStats) {
            // assume worst case
            for (int i = 0; i < bboard::AGENT_COUNT; i++) {
                bboard::AgentInfo& info = state.agents[i];
                info.bombCount = 0;
                info.bombStrength = bboard::BOARD_SIZE;
                info.canKick = true;
                info.maxBombCount = bboard::MAX_BOMBS_PER_AGENT;
            }
        }
    }
    else if (obsCopy.timeStep > 0 && this->trackStats) {
        // track agent stats from previous step
        obsCopy.TrackStats(state);
    }
    
    if (this->useVirtualStep && this->hasValidState && obsCopy.timeStep > 0){
        // perform virtual step
        obsCopy.VirtualStep(this->state, true, true, nullptr);
    }
    else {
        // convert the observation to a state
        obsCopy.ToState(state);
    }

    // std::cout << "Reconstructed State t = " << state.timeStep << std::endl;
    // state.Print();

    this->hasValidState = true;
    this->hasTrueState = false;

    _init_search(*this);
}

void PommermanState::set_agent_observation_params(const bboard::ObservationParameters params)
{
    this->agentObsParams = params;
}

void PommermanState::set_opponent_observation_params(const bboard::ObservationParameters params)
{
    this->opponentObsParams = params;
}

void PommermanState::set_planning_agents(const std::array<Clonable<bboard::Agent>*, bboard::AGENT_COUNT> agents)
{
    hasPlanningAgents = false;
    for (size_t i = 0; i < agents.size(); i++) {
        // skip own id, as we won't use this agent
        if (i == agentID) {
            continue;
        }

        Clonable<bboard::Agent>* agent = agents[i];
        if (agent != nullptr) {
            // create new clonable agent from this one and set its id
            planningAgents[i] = agent->clone();
            planningAgents[i]->get()->id = i;
            // we have at least one agent
            hasPlanningAgents = true;
        }
    }
}

void PommermanState::set_planning_agent(std::unique_ptr<Clonable<bboard::Agent>> agent, int index)
{
    // skip own id, as we won't use this agent anyway
    if (index == agentID) {
        return;
    }

    if (agent) {
        agent->get()->id = index;
    }
    planningAgents[index] = std::move(agent);

    hasPlanningAgents = false;
    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        if (planningAgents[i]) {
            hasPlanningAgents = true;
            break;
        }
    }
}

void PommermanState::planning_agents_reset()
{
    for (size_t i = 0; i < planningAgents.size(); i++) {
        if (i == agentID) {
            continue;
        }

        Clonable<bboard::Agent>* agent = planningAgents[i].get();
        if (agent != nullptr) {
            agent->get()->reset();
        }
    }

    hasBufferedActions = false;
}

void PommermanState::planning_agents_act()
{
    // we don't have to act when the actions are already buffered
    if (hasBufferedActions)
        return;

    bboard::Observation obs;
    for (size_t i = 0; i < planningAgents.size(); i++) {
        if (i == agentID) {
            continue;
        }

        if (state.agents[i].dead || !state.agents[i].visible) {
            moves[i] = bboard::Move::IDLE;
            continue;
        }

        Clonable<bboard::Agent>* agent = planningAgents[i].get();
        if (agent != nullptr) {
            bboard::Observation::Get(state, i, this->opponentObsParams, obs);
            moves[i] = agent->get()->act(&obs);
        }
    }

    hasBufferedActions = true;
}

// State methods

std::vector<Action> PommermanState::legal_actions() const
{
    // select the agent from which we are viewing this game
    const bboard::AgentInfo& self = state.agents[get_turn_agent_id()];

    std::vector<Action> legalActions;

    // it's always possible to idle
    legalActions.push_back(Action(bboard::Move::IDLE));
    // agents can only place bombs when max bomb count is not reached yet and they don't already stand on a bomb
    if (self.bombCount < self.maxBombCount && !state.HasBomb(self.x, self.y)) {
        legalActions.push_back(Action(bboard::Move::BOMB));
    }
    // check if movement is possible
    static const bboard::Move directions[4] = {bboard::Move::UP, bboard::Move::DOWN, bboard::Move::RIGHT, bboard::Move::LEFT};
    for (bboard::Move dir : directions) {
        bboard::Position dest = bboard::util::DesiredPosition(self.x, self.y, dir);
        if (bboard::util::IsOutOfBounds(dest)) {
            continue;
        }

        // check if the item at the destination is passable (and don't walk into flames)
        int destItem = state.items[dest.y][dest.x];
        if (destItem == bboard::Item::PASSAGE || bboard::IS_POWERUP(destItem)
                || (destItem == bboard::Item::BOMB && self.canKick)) {
            legalActions.push_back(Action(dir));
        }
        else if (bboard::IS_FLAME(destItem)) {
            int cumulativeTimeLeft = 0;

            // check if this flame disappears in the next step
            for (int i = 0; i < state.flames.count; i++) {
                const bboard::Flame& flame = state.flames[i];
                cumulativeTimeLeft += flame.timeLeft;

                // flame does not disappear in the next step as
                // cumulative time left is too high and we did not
                // find it in the previous flames
                if (cumulativeTimeLeft > 1) {
                    break;
                }

                // check if this is the flame we are looking for
                if (flame.position.x == dest.x && flame.position.y == dest.y) {
                    legalActions.push_back(Action(dir));
                    break;
                }
            }
        }
    }

    return legalActions;
}

void PommermanState::set(const std::string &fenStr, bool isChess960, int variant)
{
    // TODO
}

void PommermanState::get_state_planes(bool normalize, float *inputPlanes, Version version) const
{
    int turnAgentID = get_turn_agent_id();

    bboard::Observation obs;
    bboard::ObservationParameters obsParams = this->agentObsParams;
    if (this->useVirtualStep){
        obsParams.agentPartialMapView = false;
    }
    bboard::Observation::Get(state, turnAgentID, obsParams, obs);
    BoardToPlanes(&obs, turnAgentID, inputPlanes);

    if (this->statefulModel)
    {
        // add auxiliary outputs
        uint observationSize = PLANE_COUNT * PLANE_SIZE * PLANE_SIZE;

        // state input directly starts after observations
        float* statePointer = &inputPlanes[observationSize];
        uint stateSize = StateConstantsPommerman::AUXILIARY_STATE_SIZE();

        if (state.timeStep == 0)
        {
            // auxillary outputs are not filled yet => start with empty state
            std::fill_n(statePointer, stateSize, 0.0f);
        }
        else
        {
            // use the last auxiliary outputs as an input for the next state
            uint stateBegin = StateConstantsPommerman::AUXILIARY_STATE_BEGIN();
            std::copy_n(auxiliaryOutputs.begin() + stateBegin, stateSize, statePointer);
        }
    }
}

unsigned int PommermanState::steps_from_null() const
{
    return state.timeStep;
}

bool PommermanState::is_chess960() const
{
    return false;
}

std::string PommermanState::fen() const
{
    return "<fen-placeholder>";
}

bool _attribute_changed(const bboard::AgentInfo& oldInfo, const bboard::AgentInfo& newInfo) {
    return oldInfo.canKick != newInfo.canKick
            || oldInfo.bombCount != newInfo.bombCount
            || oldInfo.bombStrength != newInfo.bombStrength
            || oldInfo.maxBombCount != newInfo.maxBombCount;
}

void PommermanState::do_action(Action action)
{
    if (hasPlanningAgents) {
        // fill the remaining moves
        planning_agents_act();
    }

    int turnAgentID = get_turn_agent_id();

#ifdef MCTS_SINGLE_PLAYER
    // set the own action
    moves[turnAgentID] = bboard::Move(action);
#else
    // we first let our own agent move, then the opponent
    // self (buffered) -> opponent (step) -> self (buffered) -> opponent (step) ...
    moves[turnAgentID] = bboard::Move(action);
    if (myTurn) {
        // make sure that the actions survive the clone operation
        // even if we have no planning agent opponents
        hasBufferedActions = true;
        myTurn = false;
        return;
    }
    else {
        // we do the step and will continue from our own perspective
        myTurn = true;
    }
#endif
    // std::cout << "Moves: " << (int)moves[0] << " " << (int)moves[1] << " " << (int)moves[2] << " " << (int)moves[3] << std::endl;
    state.Step(moves);
    // after this step, any buffered actions are invalid
    hasBufferedActions = false;
#ifndef MCTS_SINGLE_PLAYER
    // update simulated opponent ID
    simulatedOpponentID = _get_closest_opponent(&state, agentID);
#endif
}

void PommermanState::undo_action(Action action) {
    // TODO
}

void PommermanState::prepare_action()
{
    if (hasPlanningAgents) {
        // buffer actions instead of calculcating them
        // each time we execute do_action (= expand the node)
        planning_agents_act();
    }
}


unsigned int PommermanState::number_repetitions() const
{
    return 0;
}

int PommermanState::side_to_move() const
{
    return agentID;
}

Key PommermanState::hash_key() const
{
    const bboard::AgentInfo& self = state.agents[agentID];
    int pos = self.x + self.y * bboard::BOARD_SIZE;
    // pos is in range [0, 120], so we can shift left by 7 (128)
    return ((Key)eventHash << 7) + pos;
}

void PommermanState::flip()
{
    // pass
}

Action PommermanState::uci_to_action(std::string &uciStr) const
{
    // TODO
    return Action(bboard::Move::IDLE);
}

std::string PommermanState::action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin, bool bookMove) const
{
    // TODO: Maybe change to UTF8 symbols later
    switch(bboard::Move(action)) {
    case (bboard::Move::IDLE):
        return "I";
    case (bboard::Move::UP):
        return "U";
    case (bboard::Move::DOWN):
        return "D";
    case (bboard::Move::LEFT):
        return "L";
    case (bboard::Move::RIGHT):
        return "R";
    case (bboard::Move::BOMB):
        return "B";
    default:
        return "?";
    }
}

inline TerminalType is_terminal_v1(const PommermanState* pommerState, size_t numberLegalMoves, float& customTerminalValue)
{
#ifndef MCTS_SINGLE_PLAYER
    if(!pommerState->myTurn) {
        // the other agent is always at intermediate steps
        // and we never want to stop the search there
        customTerminalValue = 0.0f;
        return TERMINAL_NONE;
    }
#endif

    int turnAgentID = pommerState->get_turn_agent_id();
    const bboard::State& state = pommerState->state;

    if(state.finished)
    {
        if(state.IsWinner(turnAgentID))
        {
            customTerminalValue = 1.0f;
            return TERMINAL_WIN;
        }
        else
        {
            if(state.isDraw || state.timeStep > 800)
            {
                customTerminalValue = 0.0f;
                return TERMINAL_DRAW;
            }
            else
            {
                customTerminalValue = -1.0f;
                return TERMINAL_LOSS;
            }
        }
    }

    // state is not finished
    if(state.agents[turnAgentID].dead)
    {
        if (pommerState->gameMode == bboard::GameMode::FreeForAll) {
            customTerminalValue = -1.0f;
            return TERMINAL_LOSS;
        }
        // Partner is still alive
        // TODO: Add evaluation from NN
        customTerminalValue = -0.5f;
        return TERMINAL_CUSTOM;
    }

    return TERMINAL_NONE;
}

#ifndef MCTS_SINGLE_PLAYER
inline TerminalType is_terminal_ffa_1vs1_sim(const PommermanState* pommerState, size_t numberLegalMoves, float& customTerminalValue)
{
    int agentID = pommerState->agentID;
    int simulatedOpponentID = pommerState->simulatedOpponentID;
    const bboard::State& state = pommerState->state;

    if(!pommerState->myTurn) {
        // the other agent is always at intermediate steps
        // and we never want to stop the search there
        customTerminalValue = 0.0f;
        return TERMINAL_NONE;
    }

    if(state.finished && state.isDraw) {
        customTerminalValue = 0.0f;
        return TERMINAL_DRAW;
    }

    // the agent always looses if it is dead
    if(state.agents[agentID].dead) {
        customTerminalValue = -1.0f;
        return TERMINAL_LOSS;
    }

    // it is considered a draw if the simulated opponent is dead => we stop the simulation
    if(state.agents[simulatedOpponentID].dead) {
        customTerminalValue = 0.0f;
        return TERMINAL_DRAW;
    }

    // the game is done
    if(state.finished) {
        if (state.IsWinner(agentID)) {
            customTerminalValue = 1.0f;
            return TERMINAL_WIN;
        }
        else {
            customTerminalValue = 0.0f;
            return TERMINAL_DRAW;
        }
    }

    return TERMINAL_NONE;
}
#endif

inline int _get_num_of_dead_opponents(const bboard::State& state, const uint ownId)
{
    const bboard::AgentInfo& ownInfo = state.agents[ownId];
    int deadOpponents = 0;
    for (uint i = 0; i < bboard::AGENT_COUNT; i++) {
        if (i == ownId) {
            continue;
        }

        const bboard::AgentInfo& info = state.agents[i];
        if (info.dead && (ownInfo.team == 0 || info.team == 0 || ownInfo.team != info.team)) {
            deadOpponents++;
        }
    }

    return deadOpponents;
}

inline TerminalType is_terminal_v2(const PommermanState* pommerState, size_t numberLegalMoves, float& customTerminalValue)
{
    const bboard::State& state = pommerState->state;
    const bboard::AgentInfo& ownInfo = state.agents[pommerState->agentID];

    // new return values
    if(state.finished || ownInfo.dead || state.timeStep >= pommerState->maxTimeStep)
    {
        int numDeadOpponents = _get_num_of_dead_opponents(state, pommerState->agentID);
        switch (pommerState->gameMode)
        {
        case bboard::GameMode::FreeForAll:
            customTerminalValue = numDeadOpponents * 1.0 / 3 + (ownInfo.dead ? -1.0 : 0.0);
            break;
        
        case bboard::GameMode::TwoTeams:
            customTerminalValue = numDeadOpponents * 1.0 / 2 + (ownInfo.dead ? -1.0 / 2 : 0.0);
            break;

        default:
            throw std::runtime_error("GameMode not supported.");
        }
        return TERMINAL_CUSTOM;
    }
    
    return TERMINAL_NONE;
}

inline TerminalType is_terminal_v4(const PommermanState* pommerState, size_t numberLegalMoves, float& customTerminalValue)
{
    const bboard::State& state = pommerState->state;
    const bboard::AgentInfo& ownInfo = state.agents[pommerState->agentID];

    // new return values
    if(state.finished || ownInfo.dead || state.timeStep >= pommerState->maxTimeStep)
    {
        int numDeadOpponents = _get_num_of_dead_opponents(state, pommerState->agentID);
        switch (pommerState->gameMode)
        {
        case bboard::GameMode::FreeForAll:
            customTerminalValue = -1 + 4.0 / 7 * numDeadOpponents + (ownInfo.dead ? 0 : 2.0 / 7);
            break;
        
        default:
            throw std::runtime_error("GameMode not supported.");
        }
        return TERMINAL_CUSTOM;
    }
    
    return TERMINAL_NONE;
}


TerminalType PommermanState::is_terminal(size_t numberLegalMoves, float& customTerminalValue) const
{
    return is_terminal_v1(this, numberLegalMoves, customTerminalValue);
}

bool PommermanState::gives_check(Action action) const
{
    return false;
}

PommermanState* PommermanState::clone() const
{
    PommermanState* clone = new PommermanState(gameMode, statefulModel, maxTimeStep);
    clone->state = state;
    clone->hasTrueState = hasTrueState;
    clone->hasValidState = hasValidState;
    clone->agentID = agentID;
#ifndef MCTS_SINGLE_PLAYER
    clone->simulatedOpponentID = simulatedOpponentID;
    clone->myTurn = myTurn;
#endif
    clone->agentObsParams = agentObsParams;
    clone->opponentObsParams = opponentObsParams;
    clone->useVirtualStep = useVirtualStep;
    clone->trackStats = trackStats;
    if (hasPlanningAgents) {
        // clone all relevant agents
        for (size_t i = 0; i < planningAgents.size(); i++) {
            if (i == agentID || state.agents[i].dead) {
                continue;
            }
            auto ptr = planningAgents[i].get();
            if (ptr != nullptr) {
                clone->planningAgents[i] = ptr->clone();
                clone->hasPlanningAgents = true;
            }
        }
    }
    clone->hasBufferedActions = hasBufferedActions;
    if (hasBufferedActions) {
        std::copy_n(moves, bboard::AGENT_COUNT, clone->moves);
    }
    if (StateConstantsPommerman::NB_AUXILIARY_OUTPUTS() != 0) {
        clone->auxiliaryOutputs = auxiliaryOutputs;  // deep copy auxiliary outputs
    }
    return clone;
}

void PommermanState::init(int variant, bool isChess960)
{
    // TODO
}

GamePhase PommermanState::get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const
{        
    const bboard::AgentInfo& self = state.agents[agentID];
    int closestDistance = bboard::BOARD_SIZE * 2;
    
    


    switch (gamePhaseDefinition)
    {
    case NONE:
        return 0;
    case MIXEDNESS:
        for (int i = 0; i < bboard::AGENT_COUNT; i++) {
            if (i == agentID) {
                continue;
            }
            const bboard::AgentInfo& other = state.agents[i];
            if (other.dead || !self.IsEnemy(other)) {
                continue;
            }

            int manhattan = abs(self.x - other.x) + abs(self.y - other.y);
            if (manhattan < closestDistance) {
                closestDistance = manhattan;
            }
        }

        if (closestDistance < 3) return 2;
        if (closestDistance < 6) return 1;
        return 0;
        break;
        
    case STEPS:
        if (state.timeStep < 40 ) return 0;
        return 1;
        break;

    case LIVING_OPPONENTS:
        return 4 - state.aliveAgents;
        break;

    default:
        break;
    }

    return -1;
    
}

void PommermanState::print(std::ostream& os) const
{
    // TODO
    os << InitialStateToString(state);
}

Tablebase::WDLScore PommermanState::check_for_tablebase_wdl(Tablebase::ProbeState& result)
{
    result = Tablebase::FAIL;
    return Tablebase::WDLScoreNone;
}

void PommermanState::set_auxiliary_outputs(const float *auxiliaryOutputs)
{
    if (StateConstantsPommerman::NB_AUXILIARY_OUTPUTS() != 0) {
        std::copy(auxiliaryOutputs, auxiliaryOutputs+StateConstantsPommerman::NB_AUXILIARY_OUTPUTS(), this->auxiliaryOutputs.begin());
    }
}
