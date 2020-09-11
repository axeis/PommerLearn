/*
 * @file: pommermanstate.cpp
 * Created on 15.07.2020
 * @author: queensgambit
 */

#include "pommermanstate.h"
#include "data_representation.h"
#include "agents.hpp"


PommermanState::PommermanState():
    agentToMove(0),
    plies(0)
{
    agentActions = new bboard::Move[numberAgents];
    for (size_t idx = 0; idx < numberAgents; ++idx) {
        agentActions[idx] = bboard::Move::IDLE;
    }
}

PommermanState::~PommermanState()
{
    delete agentActions;
}

void PommermanState::set_state(const bboard::State *state)
{
    this->state = state;
}

std::vector<Action> PommermanState::legal_actions() const
{
    return {Action(bboard::Move::IDLE),
            Action(bboard::Move::UP),
            Action(bboard::Move::LEFT),
            Action(bboard::Move::DOWN),
            Action(bboard::Move::RIGHT),
            Action(bboard::Move::BOMB)};
}

void PommermanState::set(const std::string &fenStr, bool isChess960, int variant)
{
    // TODO
}

void PommermanState::get_state_planes(bool normalize, float *inputPlanes) const
{
    // TODO
    StateToPlanes(state, 0, inputPlanes);
}

unsigned int PommermanState::steps_from_null() const
{
    return plies;
}

bool PommermanState::is_chess960() const
{
    return false;
}

std::string PommermanState::fen() const
{
    return "<fen-placeholder>";
}

void PommermanState::do_action(Action action)
{
    agentActions[agentToMove++] = bboard::Move(action);
    if (agentToMove == numberAgents) {
//        bboard::Step(state, agentActions);
        agentToMove = 0;
    }
}

void PommermanState::undo_action(Action action) {
    // TODO
}


unsigned int PommermanState::number_repetitions() const
{
    return 0;
}

int PommermanState::side_to_move() const
{
    return agentToMove;
}

Key PommermanState::hash_key() const
{
    return 0;
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
    return "";
}

TerminalType PommermanState::is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const
{
    // TODO
    return TERMINAL_NONE;
}

Result PommermanState::check_result(bool inCheck) const
{
    // TODO
}


bool PommermanState::gives_check(Action action) const
{
    return false;
}

PommermanState* PommermanState::clone() const
{
    // TODO
}

void PommermanState::print(std::ostream& os) const
{
    // TODO
    os << InitialStateToString(*state);
}

Tablebase::WDLScore PommermanState::check_for_tablebase_wdl(Tablebase::ProbeState& result)
{
    result = Tablebase::FAIL;
    return Tablebase::WDLScoreNone;
}
