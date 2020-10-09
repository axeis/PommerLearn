#ifndef RUNNER_H
#define RUNNER_H

#include "bboard.hpp"
#include "agents.hpp"
#include "ipc_manager.h"
#include "episode_info.h"

/**
 * @brief The Runner class povides utilities for running simulations in the pommerman environment.
 */
class Runner
{
public:
    Runner();
    /**
     * @brief run_episode Simulate a single episode with the given environment and maximum steps.
     * @param env The environment to use. Must be initialized.
     * @param maxSteps The maximum number of steps of the episode.
     * @param printSteps Whether to print the steps.
     * @return The result of the episode.
     */
    static EpisodeInfo run_env_episode(bboard::Environment& env, int maxSteps, bool printSteps=false);

    /**
     * @brief run Run the environment with the given agents and optionally collect logs.
     * @param agents The agents which will be used in the environment.
     * @param gameMode The gamemode of the environment.
     * @param maxEpisodeSteps The maximum number of steps per episode.
     * @param maxEpisodes The maximum number of episodes. Ignored if -1.
     * @param maxLoggedSteps The maximum total number of logged steps after this call. Starts new episodes when this limit is not reached. Ignored if -1.
     * @param seed The seed used to generate the training data (for deterministic results). Ignored (initialized with time) if -1.
     * @param printSteps Whether to print the steps.
     * @param ipcManager The IPCManager which is used to save/transmit the episode logs. No logs are saved if this is a nullptr.
     */
    static void run(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, bboard::GameMode gameMode, int maxEpisodeSteps, long maxEpisodes, long maxLoggedSteps = -1, long seed = -1, bool printSteps = false, IPCManager* ipcManager = nullptr);

    /**
     * @brief run_simple_agents Run the environment with simple agents and optionally collect logs.
     * @param maxEpisodeSteps The maximum number of steps per episode.
     * @param maxEpisodes The maximum number of episodes. Ignored if -1.
     * @param maxLoggedSteps The maximum total number of logged steps after this call. Starts new episodes when this limit is not reached. Ignored if -1.
     * @param seed The seed used to generate the training data (for deterministic results). Ignored (initialized with time) if -1.
     * @param printSteps Whether to print the steps.
     * @param ipcManager The IPCManager which is used to save/transmit the episode logs. No logs are saved if this is a nullptr.
     */
    static void run_simple_agents(int maxEpisodeSteps, long maxEpisodes, long maxLoggedSteps = -1, long seed = -1, bool printSteps = false, IPCManager* ipcManager = nullptr);

private:

};

#endif // RUNNER_H
