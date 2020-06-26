import numpy as np
from pommerman import constants


def get_agent_actions(z, episode):
    episode_steps = z.attrs['EpisodeSteps'][episode]
    actions = np.ones((episode_steps, 4)) * constants.Action.Stop.value

    act = z['act']

    episode_start = episode * 4
    offset = int(np.sum(z.attrs['AgentSteps'][0:episode_start]))
    for a in range(0, 4):
        agent_steps = z.attrs['AgentSteps'][episode_start + a]
        actions[0:agent_steps, a] = act[offset:(offset + agent_steps)]

        offset += agent_steps

    return actions


def get_agent_episode_slice(z, agent_episode):
    # sum up all steps up to our episode
    start_index = int(np.sum(z.attrs['AgentSteps'][0:agent_episode]))
    # add the amount of steps of the episode
    end_index = start_index + z.attrs['AgentSteps'][agent_episode]
    return slice(start_index, end_index)


def last_episode_is_cut(z):
    return np.sum(z.attrs.get('AgentSteps')) != z.attrs.get('Steps')
