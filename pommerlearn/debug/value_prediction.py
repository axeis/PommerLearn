from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

from torch.distributions.categorical import Categorical
from tqdm import tqdm

from dataset_util import PommerDataset, get_agent_episode_slice
import torch

from nn.PommerModel import PommerModel


def plot_value_prediction(dataset_path, model_path, agent_episode, figure_comment, save_figure, show_figure):
    z = zarr.open(str(dataset_path), 'r')
    print("Total number of episodes in dataset", len(z.attrs.get('AgentSteps')))

    data = PommerDataset.from_zarr(z, discount_factor=1)
    episode_slice = get_agent_episode_slice(z, agent_episode)
    num_steps = episode_slice.stop - episode_slice.start

    print(f"Episode {agent_episode}: Steps {num_steps} from {episode_slice.start} to {episode_slice.stop}, "
          f"val[0] {data[episode_slice.start].val.item()}")

    def get_alive_agents(simple_obs):
        agent_relative_0_alive = simple_obs[10].sum()
        agent_relative_1_alive = simple_obs[11].sum()
        agent_relative_2_alive = simple_obs[12].sum()
        agent_relative_3_alive = simple_obs[13].sum()

        return np.array([agent_relative_0_alive, agent_relative_1_alive, agent_relative_2_alive, agent_relative_3_alive])

    def add_alive_changes_to_dict(step, changes, changes_dict):
        if changes.sum() > 0:
            changes_dict[step] = []
            for a in range(0, 4):
                if changes[a] > 0:
                    changes_dict[step].append(a)

    agents_died_steps = {}
    predicted_values = np.empty(num_steps)

    model = torch.jit.load(f"{model_path}/torch_{'cuda' if torch.cuda.is_available() else 'cpu'}/model-bsize-1.pt")
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    alive_agents = None

    for i in tqdm(range(0, num_steps)):
        obs_tensor = torch.as_tensor(data[episode_slice.start + i].obs)
        # unsqueeze to add batch dimension
        value_out, policy_out = model(PommerModel.flatten(obs_tensor.unsqueeze(0).to(device=device_name), None))
        predicted_values[i] = value_out
        # print(f"Step {i - episode_slice.start}: {value_out, policy_out}")

        if alive_agents is None:
            alive_agents = get_alive_agents(obs_tensor)
        else:
            new_alive_agents = get_alive_agents(obs_tensor)
            changes = alive_agents - new_alive_agents
            add_alive_changes_to_dict(i, changes, agents_died_steps)
            alive_agents = new_alive_agents

    episode_end_dead = np.array(z.attrs.get('EpisodeDead')[agent_episode], dtype=int)
    episode_end_alive = 1 - episode_end_dead
    changes = alive_agents - episode_end_alive
    add_alive_changes_to_dict(num_steps + 1, changes, agents_died_steps)

    steps = np.arange(0, num_steps)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Value')

    real_value = data[episode_slice].val
    ax1.plot(steps, predicted_values, label="predicted values")
    ax1.plot(steps, data[episode_slice].val, label="target values")

    minv = min(predicted_values.min(), real_value.min())
    maxv = max(predicted_values.max(), real_value.max())
    for s in agents_died_steps:
        relative_agent_ids = agents_died_steps[s]
        plt.vlines(s, minv, maxv, colors="black")
        text = f"{' and '.join([str(id) for id in relative_agent_ids])} died "
        plt.annotate(text, (s, (maxv + minv) / 2), ha='right', fontsize=14)

    fig.tight_layout()
    plt.legend(loc='lower left')

    if save_figure:
        fig.savefig(f"values_{figure_comment}_{agent_episode}.svg", bbox_inches='tight')

    if show_figure:
        plt.show()


# nice test episodes in mcts_data_500:
# 484: loss (second place)
# 530: win stable
# 531: win unstable
# 532: tie between 0 and 1

plot_value_prediction(
    dataset_path="mcts_data_500.zr",
    model_path="mcts_500_new_value_g1-model",
    agent_episode=532,
    figure_comment="new_value_g1",
    save_figure=True,
    show_figure=True
)
