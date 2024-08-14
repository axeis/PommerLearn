#include "data_representation.h"

#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include <iostream>
#include <xtensor/xio.hpp>

bool CENTERED_OBSERVATION;
std::string PHASE_DEFINITION;

float inline _getNormalizedBombStrength(int stength)
{
    float val = (float)stength / bboard::BOARD_SIZE;
    return val > 1.0f ? 1.0f : val;
}

bool inline _isOutOfPlane(const bboard::AgentInfo &self, int x, int y)
{
    if (!CENTERED_OBSERVATION)
    {
        return false;
    }

    const int planeViewSize = (PLANE_SIZE - 1) / 2;
    return abs(x - self.x) > planeViewSize || abs(y - self.y) > planeViewSize;
}

template <typename xtPlanesType>
inline void _boardToPlanes(const bboard::Board *board, int id, xtPlanesType xtPlanes, int &planeIndex)
{
    // reset content
    xt::view(xtPlanes, xt::all()) = 0;

    // obstacle planes
    int rigidPlane = planeIndex++;
    int woodPlane = planeIndex++;

    // item planes
    int extraBombPlane = planeIndex++;
    int incrangePlane = planeIndex++;
    int kickPlane = planeIndex++;

    // bomb planes
    int bombTimePlane = planeIndex++;
    int bombStrengthPlane = planeIndex++;
    int bombMovementHorizontalPlane = planeIndex++;
    int bombMovementVerticalPlane = planeIndex++;

    int flamesPlane = planeIndex++;

    int agent0Plane = planeIndex++;
    int agent1Plane = planeIndex++;
    int agent2Plane = planeIndex++;
    int agent3Plane = planeIndex++;
    int agentOffset = 4 - id;

    for (int y = 0; y < bboard::BOARD_SIZE; y++)
    {
        for (int x = 0; x < bboard::BOARD_SIZE; x++)
        {
            const bboard::Item item = static_cast<bboard::Item>(board->items[y][x]);
            if (bboard::IS_WOOD(item))
            {
                xt::view(xtPlanes, woodPlane, y, x) = 1;
                continue;
            }
            switch (item)
            {
            case bboard::Item::RIGID:
            {
                xt::view(xtPlanes, rigidPlane, y, x) = 1;
                break;
            }
            case bboard::Item::EXTRABOMB:
            {
                xt::view(xtPlanes, extraBombPlane, y, x) = 1;
                break;
            }
            case bboard::Item::INCRRANGE:
            {
                xt::view(xtPlanes, incrangePlane, y, x) = 1;
                break;
            }
            case bboard::Item::KICK:
            {
                xt::view(xtPlanes, kickPlane, y, x) = 1;
                break;
            }
            case bboard::Item::AGENT0:
            {
                xt::view(xtPlanes, agent0Plane + ((0 + agentOffset) % 4), y, x) = 1;
                break;
            }
            case bboard::Item::AGENT1:
            {
                xt::view(xtPlanes, agent0Plane + ((1 + agentOffset) % 4), y, x) = 1;
                break;
            }
            case bboard::Item::AGENT2:
            {
                xt::view(xtPlanes, agent0Plane + ((2 + agentOffset) % 4), y, x) = 1;
                break;
            }
            case bboard::Item::AGENT3:
            {
                xt::view(xtPlanes, agent0Plane + ((3 + agentOffset) % 4), y, x) = 1;
                break;
            }
            default:
            {
                break;
            }
            }
        }
    }

    if (CENTERED_OBSERVATION)
    {
        // If the observation is centered, our own position is always at the center.
        // Adding this would not provide any additional information.
        // Instead, we highlight the board's boundaries.
        xt::view(xtPlanes, agent0Plane + ((0 + agentOffset) % 4)) = 1;
    }

    for (int i = 0; i < board->bombs.count; i++)
    {
        bboard::Bomb bomb = board->bombs[i];
        int x = bboard::BMB_POS_X(bomb);
        int y = bboard::BMB_POS_Y(bomb);

        // bombs explode at BMB_TIME == 0, we invert that to get values from 0->1 until the bombs explode
        xt::view(xtPlanes, bombTimePlane, y, x) = 1 - ((float)bboard::BMB_TIME(bomb) / bboard::BOMB_LIFETIME);
        xt::view(xtPlanes, bombStrengthPlane, y, x) = _getNormalizedBombStrength(bboard::BMB_STRENGTH(bomb));

        // bomb movement
        bboard::Move bombMovement = bboard::Move(bboard::BMB_DIR(bomb));
        switch (bombMovement)
        {
        case bboard::Move::UP:
            xt::view(xtPlanes, bombMovementVerticalPlane, y, x) = 1.0f;
            break;
        case bboard::Move::DOWN:
            xt::view(xtPlanes, bombMovementVerticalPlane, y, x) = -1.0f;
            break;
        case bboard::Move::LEFT:
            xt::view(xtPlanes, bombMovementHorizontalPlane, y, x) = -1.0f;
            break;
        case bboard::Move::RIGHT:
            xt::view(xtPlanes, bombMovementHorizontalPlane, y, x) = 1.0f;
            break;

        default:
            break;
        }
    }

    // flame plane (lifetime)
    float cumulativeTimeLeft = 0;
    for (int i = 0; i < board->flames.count; i++)
    {
        const bboard::Flame &flame = board->flames[i];

        cumulativeTimeLeft += (float)flame.timeLeft;
        float flameValue = cumulativeTimeLeft / bboard::FLAME_LIFETIME;
        xt::view(xtPlanes, flamesPlane, flame.position.y, flame.position.x) = flameValue;
    }
}

template <typename xtPlanesType>
inline void _infoToPlanes(const bboard::AgentInfo *info, xtPlanesType xtPlanes, int &planeIndex)
{
    xt::view(xtPlanes, planeIndex++) = _getNormalizedBombStrength(info->bombStrength);
    xt::view(xtPlanes, planeIndex++) = (float)info->bombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = (float)info->maxBombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = info->canKick ? 1.0f : 0.0f;
}

template <typename xtPlanesType>
inline void _aliveToPlanes(const bboard::Board *board, const int id, xtPlanesType xtPlanes, int &planeIndex)
{
    for (int i = 0; i < bboard::AGENT_COUNT; i++)
    {
        xt::view(xtPlanes, planeIndex++) = board->agents[(id + i) % bboard::AGENT_COUNT].dead ? 0.0f : 1.0f;
    }
}

template <typename xtPlanesType>
inline void _shiftPlanes(const bboard::Board *board, int id, xtPlanesType xtPlanes)
{

    // move values appearing in agents view & set
    // 1|2|3            1|4|5               0|4|5
    // 4|5|6    -->     4|x|8       -->     0|x|8
    // x|8|9            x|8|9               0|0|0
    const int n = bboard::BOARD_SIZE;
    int shiftX = board->agents[id].x - (n >> 1);
    int shiftY = board->agents[id].y - (n >> 1);
    auto destY = xt::range(std::max(0, -shiftY), n - std::max(0, shiftY));
    auto destX = xt::range(std::max(0, -shiftX), n - std::max(0, shiftX));
    auto srcY = xt::range(std::max(0, shiftY), n + std::min(0, shiftY));
    auto srcX = xt::range(std::max(0, shiftX), n + std::min(0, shiftX));
    auto planeRange = xt::range(0, N_POSITION_DEPENDENT_PLANES);

    xt::view(xtPlanes, planeRange, destY, destX) = xt::view(xtPlanes, planeRange, srcY, srcX);

    if (shiftX < 0)
    {
        xt::view(xtPlanes, planeRange, xt::all(), xt::range(0, abs(shiftX))) = 0;
    }
    else
    {
        xt::view(xtPlanes, planeRange, xt::all(), xt::range(n - shiftX, n)) = 0;
    }

    if (shiftY < 0)
    {
        xt::view(xtPlanes, planeRange, xt::range(0, abs(shiftY)), xt::all()) = 0;
    }
    else
    {
        xt::view(xtPlanes, planeRange, xt::range(n - shiftY, n), xt::all()) = 0;
    }
}

/**
 * @brief
 *
 * @param board the Board to be processed
 * @param id the ID of the Agent
 * @param planes Planes to wirte the result to
 */
void BoardToPlanes(const bboard::Board *board, int id, float *planes)
{
    // shape of all planes of a state
    std::vector<std::size_t> stateShape = {PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
    auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

    int planeIndex = 0;
    _boardToPlanes(board, id, xtPlanes, planeIndex);
    _infoToPlanes(&board->agents[id], xtPlanes, planeIndex);
    _aliveToPlanes(board, id, xtPlanes, planeIndex);
    xt::view(xtPlanes, planeIndex++) = (float)board->timeStep / 799.0f;

    if (CENTERED_OBSERVATION)
    {
        _shiftPlanes(board, id, xtPlanes);
    }
}

int8_t GetPhase(const float *planes)
{
    if (PHASE_DEFINITION == "steps")
    {
        if ((int)(planes[PLANES_TOTAL_FLOATS - 1] * 799.0f) <= 40)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
    else if (PHASE_DEFINITION == "mixedness")
    {
        // todo: differentiate between ffa and team
        std::vector<std::size_t> stateShape = {PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
        auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

        std::vector<std::pair<std::size_t, std::size_t>> indices;

        for (int i = 0; i < 3; i++)
        {
            auto view = xt::view(xtPlanes, 11+i, xt::all(), xt::all());
            auto opponentView = xt::reshape_view(view, {11, 11});
            //Todo: das ausprobieren
            // auto opponentView = xt::squeeze(view, {0});
            


            // auto shape = opponentView.shape();
            // std::cout << "Shape of the view: (";
            // for (std::size_t i = 0; i < shape.size(); ++i) {
            //     std::cout << shape[i];
            //     if (i < shape.size() - 1) {
            //     std::cout << ", ";
            //     }
            // }
            // std::cout << ")" << std::endl;

            size_t y;
            for (size_t x = 0; x < PLANE_SIZE; x++)
            {
                for (y = 0; y < PLANE_SIZE; y++)
                {
                    if (opponentView(0, x, y))
                    {
                        indices.emplace_back(x,y);
                        break;
                    }
                }
                if (opponentView(0, x, y))
                    break;
            }
        }
        // Check if winning 
        if (indices.size() < 1) return 2;
        
        int minManhattanDistance = PLANE_SIZE + PLANE_SIZE;
        for (size_t i = 1; i < 4; i++)
        {
            int manhattanDistance = abs(indices[0].first - indices[i].first) +
                                    abs(indices[0].second - indices[i].second); 
            if (manhattanDistance < minManhattanDistance)
            {
                minManhattanDistance = manhattanDistance;
            }                            
        }
    
        //ToDo tune cut off values
        if(minManhattanDistance < 3 ) return 2;
        if(minManhattanDistance < 6 ) return 1;
        return 0;
    }
    else if (PHASE_DEFINITION == "living_opponents")
    {
        // todo check for game mode and apply only for actual opponent in team mode
        std::vector<std::size_t> stateShape = {PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
        auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

        int livingCount = 0;

        for (int i = 0; i < 3; i++)
        {
            auto opponentSlice = xt::view(xtPlanes, xt::keep(19 + i), xt::all(), xt::all());
            if (opponentSlice(0, 0, 0) == 1)
                livingCount++;
        }

        return 3 - livingCount;
    }
}

std::string InitialStateToString(const bboard::State &state)
{
    std::stringstream stream;

    for (int y = 0; y < bboard::BOARD_SIZE; y++)
    {
        for (int x = 0; x < bboard::BOARD_SIZE; x++)
        {
            int elem = state.items[y][x];

            switch (elem)
            {
            case bboard::Item::PASSAGE:
                stream << "0";
                break;
            case bboard::Item::RIGID:
                stream << "1";
                break;
            case bboard::Item::WOOD:
                stream << "2";
                break;
            case bboard::Item::AGENT0:
                stream << "A";
                break;
            case bboard::Item::AGENT1:
                stream << "B";
                break;
            case bboard::Item::AGENT2:
                stream << "C";
                break;
            case bboard::Item::AGENT3:
                stream << "D";
                break;
            default:
                if (bboard::IS_WOOD(elem))
                {
                    int item = state.FlagItem(bboard::WOOD_POWFLAG(elem));
                    switch (item)
                    {
                    case bboard::EXTRABOMB:
                        stream << "3";
                        break;
                    case bboard::INCRRANGE:
                        stream << "4";
                        break;
                    case bboard::KICK:
                        stream << "5";
                        break;
                    // when we do not know this item, treat it like a regular wood (should not happen!)
                    default:
                        std::cerr << "Error: Encountered unknown item at (" << x << ", " << y << "): " << elem << ", item:" << item << std::endl;
                        // treat as regular wood
                        stream << "2";
                        break;
                    }
                }
                else
                {
                    std::cerr << "Error: Encountered unknown element at (" << x << ", " << y << "): " << elem << std::endl;
                    // ignore everything else
                    stream << "0";
                }

                break;
            }
        }
    }

    return stream.str();
}
