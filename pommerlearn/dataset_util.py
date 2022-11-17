import gc
import logging
from collections import namedtuple
from pathlib import Path
from typing import List, Union, Tuple, Optional, NamedTuple

import numpy as np
import torch
from pommerman import constants
import zarr
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter


class PommerSample(NamedTuple):
    """
    Holds a single sample or a batch of samples.
    """
    obs: torch.Tensor
    val: torch.Tensor
    act: torch.Tensor
    pol: torch.Tensor

    @staticmethod
    def merge(a, b):
        """
        Merges two samples and adds the batch dimension if necessary.

        :param a: A sample or batch of samples
        :param b: A sample or batch of samples
        :returns: a batch of samples containing a and b
        """
        if not a.is_batch():
            a = a.expand_batch_dim()
        if not b.is_batch():
            b = b.expand_batch_dim()

        return PommerSample(
            torch.cat((a.obs, b.obs), dim=0),
            torch.cat((a.val, b.val), dim=0),
            torch.cat((a.act, b.act), dim=0),
            torch.cat((a.pol, b.pol), dim=0)
        )

    def is_batch(self):
        return len(self.obs.shape) == 4

    def expand_batch_dim(self):
        """
        Creates a new sample with added batch dimension.

        :returns: this sample expanded by the batch dimension.
        """
        assert not self.is_batch()
        return PommerSample(
            self.obs.unsqueeze(0),
            self.val.unsqueeze(0),
            self.act.unsqueeze(0),
            self.pol.unsqueeze(0)
        )

    def batch_at(self, index):
        """
        Select a single sample from this batch according to the given index.

        :param index: The index of the sample within this batch
        :returns: a new sample with data from the specified index (no batch dimension)
        """
        assert self.is_batch()
        return PommerSample(
            self.obs[index],
            self.val[index],
            self.act[index],
            self.pol[index]
        )

    def equals(self, other):
        return self.obs.shape == other.obs.shape \
               and (self.val == other.val).all() \
               and (self.act == other.act).all() \
               and (self.pol == other.pol).all() \
               and (self.obs == other.obs).all()


class PommerDataset(Dataset):
    """
    A pommerman dataset.
    """
    PLANE_HORIZONTAL_BOMB_MOVEMENT = 7
    PLANE_VERTICAL_BOMB_MOVEMENT = 8
    PLANE_AGENT0 = 10
    PLANE_AGENT1 = 11
    PLANE_AGENT2 = 12
    PLANE_AGENT3 = 13

    def __init__(self, obs, val, act, pol, ids, steps_to_end, transform=None, sequence_length=None, return_ids=False):
        assert len(obs) == len(val) == len(act) == len(pol), \
            f"Sample array lengths are not the same! Got: {len(obs)}, {len(val)}, {len(act)}, {len(pol)}"

        def is_np_or_zarr(data):
            return isinstance(data, np.ndarray) or isinstance(obs, zarr.core.Array)

        if not is_np_or_zarr(obs) or not is_np_or_zarr(val) or not is_np_or_zarr(act) or not is_np_or_zarr(pol):
            assert False, "Invalid data type!"

        self.sequence_length = sequence_length
        self.episode = None

        if self.sequence_length is not None:
            assert self.sequence_length >= 1, "Invalid sequence length!"

        self.obs = obs
        self.val = val
        self.act = act
        self.pol = pol
        self.ids = ids
        self.steps_to_end = steps_to_end

        self.return_ids = return_ids
        self.transform = transform

    @staticmethod
    def from_zarr_path(path: Path, discount_factor: float, mcts_val_weight: Optional[float],
                       transform=None, return_ids=False, verbose: bool = False):
        z = zarr.open(str(path), 'r')
        return PommerDataset.from_zarr(z, discount_factor, mcts_val_weight, transform, return_ids,
                                       verbose)

    @staticmethod
    def from_zarr(z: zarr.Group, discount_factor: float, mcts_val_weight: Optional[float],
                  transform=None, return_ids=False, verbose: bool = False):
        if verbose:
            print(
                f"Opening dataset {str(z.path)} with {z.attrs['Steps']} samples "
                f"from {len(z.attrs['EpisodeSteps'])} episodes"
            )

        z_steps = z.attrs['Steps']
        pol = z['pol'][:z_steps]

        if not np.isfinite(pol).all():
            non_finite = ~np.isfinite(pol)
            non_finite_steps = non_finite.sum(axis=-1) > 0
            logging.warning(
                f"Your zarr dataset contains non-finite policy values.\n"
                f"Found {np.unique(pol[non_finite])}, total {non_finite_steps.sum()} "
                f"({non_finite_steps.sum() / pol.shape[0] * 100} %) of steps contain non-finite values.\n"
                f"Replaced them with (1 - sum(finite_vals)) / num_of_non_finite_vals."
            )

            # remember positions of finite values
            pol_steps_finite_mask = np.isfinite(pol[non_finite_steps])
            # remove infinite values from policy
            new_pol = np.nan_to_num(pol[non_finite_steps], nan=0.0, posinf=0.0, neginf=0.0)
            # calculate sum of finite values (policy has to sum up to 1) and number of non-finite values
            sum_finite = (new_pol * pol_steps_finite_mask).sum(axis=-1)
            num_non_finite = (~pol_steps_finite_mask).sum(axis=-1)
            # replace infinite values with equal distribution
            new_pol += ~pol_steps_finite_mask * ((1 - sum_finite) / num_non_finite)[:, np.newaxis]
            # update policy
            pol[non_finite_steps] = new_pol

        return PommerDataset(
            obs=z['obs'][:z_steps],
            val=get_value_target(z, discount_factor, mcts_val_weight),
            act=z['act'][:z_steps],
            pol=pol,
            ids=get_unique_agent_episode_id(z),
            transform=transform,
            return_ids=return_ids,
            steps_to_end=get_steps_until_end(z)
        )

    @staticmethod
    def create_empty(count, transform=None, sequence_length=None, return_ids=False):
        """
        Create an empty sample container.

        :param count: The number of samples
        """

        return PommerDataset(
            obs=np.empty((count, 18, 11, 11), dtype=np.single),
            val=np.empty(count, dtype=np.single),
            act=np.empty(count, dtype=np.byte),
            pol=np.empty((count, 6), dtype=np.single),
            ids=np.empty(count, dtype=np.int32),
            steps_to_end=np.empty(count, dtype=np.int16),
            transform=transform,
            sequence_length=sequence_length,
            return_ids=return_ids
        )

    def set(self, other_samples, to_index, from_index=0, count: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Sets own_samples[to_index:to_index + count] = other_samples[from_index:from_index + count].

        :param other_samples: The other samples which should overwrite samples in this sample object
        :param to_index: The destination index
        :param from_index: The source index
        :param count: The number of elements which will be copied. If None, all samples will be copied.
        :param batch_size: The maximal number of batch_size elements that are copied simultaneously
        """

        max_count = len(other_samples) - from_index
        if count is None:
            count = max_count
        elif count > max_count:
            logging.warning(f"Source dataset holds {len(other_samples)} samples. Cannot copy {count} elements starting "
                            f"at {from_index}. Changed this to {max_count}.")
            count = max_count
        count = len(other_samples) - from_index if count is None else count
        # tuples that define what has to be copied (to_start, to_end, from_start, from_end)
        copy_tuples = []
        if batch_size is None:
            # just copy everything at once
            copy_tuples.append((to_index, to_index + count, from_index, from_index + count))
        else:
            # split whole interval with "count" elements into multiple copy instructions that copy at most
            # "batch_size" elements at once
            offset = 0
            while offset < count:
                num_el = min(batch_size, count - offset)
                idx_to = to_index + offset
                idx_from = from_index + offset

                copy_tuples.append((idx_to, idx_to + num_el, idx_from, idx_from + num_el))

                offset += num_el

        for (to_start, to_end, from_start, from_end) in copy_tuples:
            self.obs[to_start:to_end] = other_samples.obs[from_start:from_end]
            self.val[to_start:to_end] = other_samples.val[from_start:from_end]
            self.act[to_start:to_end] = other_samples.act[from_start:from_end]
            self.pol[to_start:to_end] = other_samples.pol[from_start:from_end]
            self.ids[to_start:to_end] = other_samples.ids[from_start:from_end]
            self.steps_to_end[to_start:to_end] = other_samples.steps_to_end[from_start:from_end]

            # explicity collect garbage in batch_wise transfer
            if len(copy_tuples) > 1:
                gc.collect()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.sequence_length is None:
            # return a single sample

            sample = PommerSample(
                torch.tensor(self.obs[idx], dtype=torch.float),
                torch.tensor(self.val[idx], dtype=torch.float),
                torch.tensor(self.act[idx], dtype=torch.int),
                torch.tensor(self.pol[idx], dtype=torch.float),
            )

            if self.transform is not None:
                sample = self.transform(sample)

            if self.return_ids:
                return (self.ids[idx], *sample)
            else:
                return sample
        else:
            # build a sequence of samples
            sequence = PommerSample(
                torch.zeros((self.sequence_length, 18, 11, 11), dtype=torch.float),
                torch.zeros(self.sequence_length, dtype=torch.float),
                torch.zeros(self.sequence_length, dtype=torch.int),
                torch.zeros((self.sequence_length, 6), dtype=torch.float)
            )

            # check if we have to stop before sequence_length samples
            current_id = self.ids[idx]
            end_idx = idx
            for seq_idx in range(1, self.sequence_length):
                data_idx = idx + seq_idx

                if data_idx >= len(self.obs) or self.ids[data_idx] != current_id:
                    # we reached a different episode / the beginning of the dataset
                    break

                end_idx = data_idx

            # TODO: Use PackedSequence instead of manual padding?
            seq_end = end_idx - idx + 1
            sequence.obs[0:seq_end] = torch.tensor(self.obs[idx:end_idx+1], dtype=torch.float)
            sequence.val[0:seq_end] = torch.tensor(self.val[idx:end_idx+1], dtype=torch.float)
            sequence.act[0:seq_end] = torch.tensor(self.act[idx:end_idx+1], dtype=torch.int)
            sequence.pol[0:seq_end] = torch.tensor(self.pol[idx:end_idx+1], dtype=torch.float)

            if self.transform is not None:
                sequence = self.transform(sequence)

            if self.return_ids:
                ret_ids = torch.ones(self.sequence_length, dtype=torch.int, requires_grad=False) * -1
                ret_ids[0:seq_end] = current_id
                return (ret_ids, *sequence)
            else:
                return sequence

    def shuffle(self):
        """
        Shuffles the dataset in-place.
        """
        rng_state = np.random.get_state()

        def shuffle_array(a: np.ndarray):
            np.random.set_state(rng_state)
            np.random.shuffle(a)

        shuffle_array(self.obs)
        shuffle_array(self.val)
        shuffle_array(self.act)
        shuffle_array(self.pol)
        shuffle_array(self.ids)
        shuffle_array(self.steps_to_end)


def get_agent_actions(z, episode):
    episode_steps = z.attrs['EpisodeSteps'][episode]
    actions = np.ones((episode_steps, 4)) * constants.Action.Stop.value
    raw_actions = z.attrs['EpisodeActions'][episode]

    for a in range(0, 4):
        actions[0:len(raw_actions[a]), a] = raw_actions[a]

    return actions


def get_agent_episode_slice(z, agent_episode):
    # sum up all steps up to our episode
    start_index = int(np.sum(z.attrs['AgentSteps'][0:agent_episode]))
    # add the amount of steps of the episode
    end_index = start_index + z.attrs['AgentSteps'][agent_episode]
    return slice(start_index, end_index)


def last_episode_is_cut(z):
    return np.sum(z.attrs.get('AgentSteps')) != z.attrs.get('Steps')


def get_unique_agent_episode_id(z) -> np.ndarray:
    """
    Creates unique ids for every new agent episode in the environment and returns an array containing the id of each
    individual step.

    :param z: The zarr dataset
    :return: The unique agent episode ids in z
    """
    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))

    ids = np.empty(total_steps, dtype=np.int32)
    current_id = 0
    current_step = 0

    for steps in agent_steps:
        end = min(current_step + steps, total_steps)
        ids[current_step:end] = current_id
        current_step += steps
        current_id += 1

    return ids


def get_steps_until_end(z) -> np.ndarray:
    """
    Creates an array that contains the number of steps until the last step for each agent episode.

    :param z: The zarr dataset
    :return: The number of steps until the episode of this sample ends
    """
    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))

    steps_until_end = np.empty(total_steps, dtype=np.int16)
    current_step = 0

    for steps in agent_steps:
        end = min(current_step + steps, total_steps)
        # we always end at step 1 as the final state is not in the datasets
        steps_until_end[current_step:end] = np.arange(end - current_step, 0, -1)
        current_step += steps

    return steps_until_end


def get_agent_died_in_step(single_episode_actions, single_episode_dead) -> np.ndarray:
    """
    For each agent, get the step in which it died. 0 if it is still alive.

    :param single_episode_actions: The actions of all agents in this episode.
    :param single_episode_dead: The final result of the episode.
    :return: array of steps in which the agents died
    """
    died_in_step = np.empty(4, dtype=int)
    for id, actions in enumerate(single_episode_actions):
        died_in_step[id] = len(actions) if single_episode_dead[id] else 0

    return died_in_step


def get_value_target(z, discount_factor: float, mcts_val_weight: Optional[float]) -> np.ndarray:
    """
    Creates the value target for a zarr dataset z by combining episode values with value predictions from the dataset.

    :param z: The zarr dataset
    :param discount_factor: The discount factor for the episode values (not mcts values)
    :param mcts_val_weight: Static weight of mcts values (completely ignored when None)
        val_target = mcts_val_weight * mcts values + (1 - mcts_val_weight) * episode values
        Note that episode values can contain discounting towards the mcts values (or 0 if mcts_val_weight is None)
    :return: The value target for z
    """
    assert 0 <= discount_factor <= 1, f"Invalid value for discount factor {discount_factor}"
    assert mcts_val_weight is None or 0 <= mcts_val_weight <= 1, f"Invalid value for mcts value weight {mcts_val_weight}"

    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))
    agent_ids = np.array(z.attrs.get('AgentIds'))
    agent_episode = np.array(z.attrs.get('AgentEpisode'))
    episode_winner = np.array(z.attrs.get('EpisodeWinner'))
    episode_winning_team = np.array(z.attrs.get('EpisodeWinningTeam'))
    episode_dead = np.array(z.attrs.get('EpisodeDead'))
    episode_actions = z.attrs.get('EpisodeActions')
    episode_steps = np.array(z.attrs.get('EpisodeSteps'))
    episode_draw = np.array(z.attrs.get('EpisodeDraw'))
    is_ffa = episode_winning_team == -1
    # episode_done = np.array(z.attrs.get('EpisodeDone'))

    # Warning: this is not the true value of the state but instead the Q-value of the "best" (= most selected) move.
    # Alternatively, one could use z.q and get the best Q-value (max over non-nan values), but the selected Q-values
    # will probably better represent the value of the state for the actual policy (= select action according to visits).
    all_mcts_val = z.val

    if mcts_val_weight == 1:
        return all_mcts_val[:total_steps]

    def get_combined_target(mcts_val, target_val, discounting_factors):
        if mcts_val_weight is None:
            return discounting_factors * target_val

        if not np.isfinite(mcts_val).all():
            total_len = np.prod(mcts_val.shape)
            logging.warning(f"Warning: mcts_val_weight is {mcts_val_weight} but "
                            f"{total_len - np.isfinite(mcts_val).sum()} of {total_len} values in your mcts_val are not "
                            f"finite {{inf, -inf, nan}}. This destroys your target value. Could it be that your dataset"
                            f" was not created with mcts?")

        return (
            # static weight for the values from mcts
            mcts_val_weight * mcts_val
            # combine with target values
            + (1 - mcts_val_weight) * (
                # target values are discounted
                discounting_factors * target_val
                # for discounting = 0, the target is mcts_val and not 0
                + (1 - discounting_factors) * mcts_val
            )
        )

    val_target = np.empty(total_steps)
    current_step = 0
    for agent_ep_idx in range(0, len(agent_steps)):
        agent_id = agent_ids[agent_ep_idx]
        ep = agent_episode[agent_ep_idx]
        steps = agent_steps[agent_ep_idx]
        winner = episode_winner[ep]
        dead = episode_dead[ep][agent_id]

        e_draw = episode_draw[ep]
        e_steps = episode_steps[ep]

        died_in_step = get_agent_died_in_step(episode_actions[ep], episode_dead[ep])

        # min to handle cut datasets
        next_step = min(current_step + steps, total_steps)
        num_steps = next_step - current_step

        episode_discounting = np.power(discount_factor, np.arange(steps - 1, steps - 1 - num_steps, -1))
        episode_mcts_val = all_mcts_val[current_step:next_step]


        if is_ffa:
            # only distribute rewards when the (agent) episode is done
            if winner == agent_id:
                episode_value = 1
            elif e_draw and e_steps == steps:
                # agent is part of the draw
                episode_value = 0
            elif dead:
                episode_value = -1
            else:
                # episode not done and agent not dead
                episode_value = 0
        else:
            if episode_winning_team == agent_id % 2 + 1:
                episode_value = 1
            elif episode_winning_team == 0:
                episode_value = 0
            else:
                episode_value = -1


        episode_target = get_combined_target(episode_mcts_val, episode_value, episode_discounting)

        # calculate discount factors backwards
        val_target[current_step:next_step] = episode_target
        current_step = next_step

    return val_target


def split_arr_based_on_idx(array: np.ndarray, split_idx: int) -> (np.ndarray, np.ndarray):
    """
    Splits a given array into two halves at the given split index and returns the first and second half
    :param array: Array to be split
    :param split_idx: Index where to split
    :return: 1st half, 2nd half after the split
    """
    return array[:split_idx], array[split_idx:]


def get_last_dataset_path(path_infos: List[Union[str, Tuple[str, float]]]) -> str:
    """
    Returns the last path provided in the path_infos list.

    :param path_infos: The path information of the zarr datasets which should be used. Expects a single path or a list
                  containing strings (paths) or a tuple of the form (path, proportion) where 0 <= proportion <= 1
                  is the number of samples which will be selected randomly from this data set.
    :return: The last path
    """
    if isinstance(path_infos, str):
        return path_infos

    last_info = path_infos[len(path_infos) - 1]

    if isinstance(last_info, tuple):
        path, proportion = last_info
        return path
    else:
        return last_info


def create_data_loaders(path_infos: Union[str, List[Union[str, Tuple[str, float]]]],
                        discount_factor: float, mcts_val_weight: Optional[float], test_size: float, batch_size: int,
                        batch_size_test: int, train_transform = None, verbose: bool = True, sequence_length=None,
                        num_workers=2, only_test_last=False, train_sampling_mode: str = 'complete',
                        test_split_mode='simple'
                        ) -> Tuple[DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given path

    :param path_infos: The path information of the zarr datasets which should be used. Expects a single path or a list
                      containing strings (paths) or a tuple of the form (path, proportion) where 0 <= proportion <= 1
                      is the number of samples which will be selected randomly from this data set.
    :param discount_factor: The discount factor that should be used
    :param mcts_val_weight: Weight for mcts values (None if only episode values should be used)
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :param batch_size_test: Batch size to use for testing
    :param train_transform: Data transformation for train data loading
    :param verbose: Log debug information
    :param sequence_length: Sequence length used in the train loader
    :param num_workers: The number of workers used for loading
    :param only_test_last: Whether only the last dataset is used for testing
    :param train_sampling_mode: Defines how the samples are chosen. Possible values: <br>
        <list>
        <li>'complete' to load all samples in random order.</li>
        <li>'weighted_steps_to_end' to assign exponentially decreasing weights to each sample based on the
        number of steps until the individual episode ends. Samples are chosen with replacement using the
        the normalized weights as probabilities.</li>
        <li>'weighted_actions' to sample all actions with equal probability</li>
        <li>'weighted_value_class' to sample all values with equal probability (warning: not recommended when used with
        discounting & mcts as there will be too many different value classes</li>
        </list>
    :param test_split_mode: Defines how the test samples are chosen. Possible values: <br>
        <list>
        <li>'simple' to split train & test set as they are. This means the test set will contain sequential samples
            from the last episodes in the data set. </li>
        <li>'random' to shuffle the dataset before splitting. The test set will contain random samples from different
            episodes in the data set.</li>
        </list>
    :return: Training loader, validation loader
    """

    assert 0 <= test_size < 1, f"Incorrect test size: {test_size}"

    if isinstance(path_infos, str):
        path_infos = [path_infos]

    def get_elems(info):
        if isinstance(info, tuple):
            return info
        else:
            return info, 1

    def get_test_size(path_index):
        if only_test_last:
            if path_index == len(path_infos) - 1:
                return test_size
            else:
                return 0
        else:
            return test_size

    def get_total_sample_data():
        all_train_samples = 0
        all_test_samples = 0
        train_start_ixs = np.zeros(len(path_infos), dtype=int)
        test_start_ixs = np.zeros(len(path_infos), dtype=int)

        for i, info in enumerate(path_infos):
            path, proportion = get_elems(info)
            z = zarr.open(str(path), 'r')
            num_samples = int(z.attrs['Steps'] * proportion)

            test_samples = int(num_samples * get_test_size(i))
            train_samples = num_samples - test_samples

            agent_steps_cumulative = np.cumsum(np.array(z.attrs['AgentSteps']))

            if proportion < 1:
                train_start_ixs[i] = np.random.randint(0, z.attrs['Steps'] - num_samples)
                test_start_ixs[i] = z.attrs['Steps'] - test_samples
            else:
                train_start_ixs[i] = 0
                # split train/test data between two episodes
                test_start_ixs[i] = agent_steps_cumulative[np.where(agent_steps_cumulative < train_start_ixs[i] + train_samples)[0][-1]]
                # correct train & test samples
                train_samples = test_start_ixs[i] - train_start_ixs[i]
                test_samples = num_samples - train_samples

            all_train_samples += train_samples
            all_test_samples += test_samples

        return all_train_samples, all_test_samples, train_start_ixs, test_start_ixs

    total_train_samples, total_test_samples, train_start_ixs, test_start_ixs = get_total_sample_data()

    if verbose:
        print(f"Loading {total_train_samples + total_test_samples} samples from {len(path_infos)} dataset(s) with "
              f"test size {total_test_samples/(total_train_samples + total_test_samples):.2f}{' only last' if only_test_last else ''} ({total_test_samples} samples) and "
              f"split mode '{test_split_mode}'")

    data_train = PommerDataset.create_empty(total_train_samples, transform=train_transform,
                                            sequence_length=sequence_length, return_ids=(sequence_length is not None))
    data_test = PommerDataset.create_empty(total_test_samples, return_ids=(sequence_length is not None))

    # create a container for all samples
    if verbose:
        print(f"Created containers with (train: {len(data_train)}, test: {len(data_test)}) samples "
              f"and train sequence length {sequence_length}")

    buffer_train_idx = 0
    buffer_test_idx = 0
    for i, info in enumerate(path_infos):
        path, proportion = get_elems(info)
        elem_samples = PommerDataset.from_zarr_path(path, discount_factor, mcts_val_weight,
                                                    verbose=verbose)

        if verbose:
            print(f"> Loading '{path}' with proportion {proportion}")

        assert 0 <= proportion <= 1, f"Invalid proportion {proportion}"

        if test_split_mode == 'random':
            elem_samples.shuffle()
        elif test_split_mode == 'simple':
            # nothing to do
            pass
        else:
            raise ValueError(f"Unknown test split mode '{test_split_mode}'.")

        if proportion < 1:
            elem_samples_nb = int(len(elem_samples) * proportion)
            test_nb = int(elem_samples_nb * get_test_size(i))
            train_nb = elem_samples_nb - test_nb
            if verbose:
                print(f"Selected slices "
                      f"[{train_start_ixs[i]}:{train_start_ixs[i] + train_nb}] and "
                      f"[{test_start_ixs[i]}:{test_start_ixs[i] + test_nb}] "
                      f"({train_nb} + {test_nb} = {elem_samples_nb} samples)")
        else:
            elem_samples_nb = len(elem_samples)
            train_nb = test_start_ixs[i] - train_start_ixs[i]
            test_nb = elem_samples_nb - train_nb

        data_train.set(elem_samples, buffer_train_idx, train_start_ixs[i], train_nb, batch_size=10000)
        data_test.set(elem_samples, buffer_test_idx, test_start_ixs[i], test_nb, batch_size=10000)
        del elem_samples
        gc.collect()

        # copy first num_samples samples
        if verbose:
            print("Copied {} ({}, {}) samples ({:.2f}%) to buffers @ ({}, {})"
                  .format(elem_samples_nb, train_nb, test_nb, proportion * 100, buffer_train_idx, buffer_test_idx))

        buffer_train_idx += train_nb
        buffer_test_idx += test_nb

    assert buffer_train_idx == total_train_samples and buffer_test_idx == total_test_samples, \
        f"The number of copied samples is wrong.. " \
        f"{(buffer_train_idx, total_train_samples, buffer_test_idx, total_test_samples)}"

    if verbose:
        print(f"Creating DataLoaders with train sampling mode {train_sampling_mode}..")

    if train_sampling_mode == 'complete':
        train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    elif train_sampling_mode == 'weighted_steps_to_end':
        train_weights = np.clip(np.power(0.97, data_train.steps_to_end - 1), 0.05, 1)
        sampler = WeightedRandomSampler(train_weights, len(data_train), replacement=True)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    elif train_sampling_mode == 'weighted_actions':
        train_weights = np.zeros_like(data_train.val, dtype=float)
        action_counts = np.array([np.sum(data_train.act == a) for a in range(0, 6)])
        non_zero_actions = (action_counts > 0).sum()

        for a in range(0, 6):
            if action_counts[a] > 0:
                train_weights[data_train.act == a] = 100000.0 / (non_zero_actions * action_counts[a])

        sampler = WeightedRandomSampler(train_weights, len(data_train), replacement=True)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    elif train_sampling_mode == 'weighted_value_class':
        unique_values = list(np.unique(data_train.val))
        value_class_counts = {}
        value_class_weights = {}
        for value in unique_values:
            num_samples = (data_train.val == value).sum()
            value_class_counts[value] = num_samples
            value_class_weights[value] = (1.0 / len(unique_values)) * (1.0 / num_samples)

        if verbose:
            print(f"Value weighting with {len(unique_values)} classes")
            print(value_class_counts)
            print(value_class_weights)

        if discount_factor != 1:
            print("Warning: Value class weighting was created for discount factor 1.")

        train_weights = np.empty(len(data_train))
        for a in range(0, len(data_train)):
            train_weights[a] = value_class_weights[data_train.val[a]]

        sampler = WeightedRandomSampler(train_weights, len(data_train), replacement=True)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        raise ValueError(f"Unknown train_sampling_mode {train_sampling_mode}")

    test_loader = DataLoader(data_test, batch_size=batch_size_test, num_workers=num_workers) if total_test_samples > 0 else None

    if verbose:
        print("done.")

    return train_loader, test_loader


def log_dataset_stats(path, log_dir, iteration):
    """
    Log dataset stats to tensorboard.

    :param path: The path of a zarr dataset
    :param log_dir: Tensorboard log_dir
    :param iteration: The iteration this dataset belongs to
    """
    z = zarr.open(str(path), 'r')

    writer = SummaryWriter(log_dir=log_dir)

    z_steps = z.attrs['Steps']
    steps = np.array(z.attrs["EpisodeSteps"])
    winners = np.array(z.attrs["EpisodeWinner"])
    dead = np.array(z.attrs["EpisodeDead"])
    done = np.array(z.attrs["EpisodeDone"])
    actions = z.attrs["EpisodeActions"]

    pol = z['pol'][:z_steps]
    pol_entropy = -np.sum(pol * np.log(pol, out=np.zeros_like(pol), where=(pol != 0)), axis=1)
    writer.add_scalar("Dataset/Policy entropy mean", pol_entropy.mean(), iteration)
    writer.add_scalar("Dataset/Policy entropy std", pol_entropy.std(), iteration)
    writer.add_text("Dataset/Policy NaN", str(np.isnan(pol).any()), iteration)

    num_episodes = len(steps)

    writer.add_scalar("Dataset/Episodes", num_episodes, iteration)
    writer.add_scalar("Dataset/Steps mean", steps.mean(), iteration)
    writer.add_scalar("Dataset/Steps std", steps.std(), iteration)

    for a in range(0, 4):
        winner_a = np.sum(winners[:] == a)
        writer.add_scalar(f"Dataset/Win ratio {a}", winner_a / num_episodes, iteration)
        alive_a = num_episodes - np.sum(dead[:, a])
        writer.add_scalar(f"Dataset/Alive ratio {a}", alive_a / num_episodes, iteration)

        actions_a = []
        episode_steps = np.empty(len(actions))
        for i, ep in enumerate(actions):
            ep_actions = ep[a]
            episode_steps[i] = len(ep_actions)
            actions_a += ep_actions

        writer.add_scalar(f"Dataset/Steps mean {a}", episode_steps.mean(), iteration)

        # TODO: Correct bin borders
        writer.add_histogram(f"Dataset/Actions {a}", np.array(actions_a), iteration)

    no_winner = np.sum((winners == -1) * (done == True))
    writer.add_scalar(f"Dataset/Draw ratio", no_winner / num_episodes, iteration)

    not_done = np.sum(done == False)
    writer.add_scalar(f"Dataset/Not done ratio", not_done / num_episodes, iteration)

    writer.close()
