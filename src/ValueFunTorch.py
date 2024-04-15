from ast import Dict, List

# from msilib.schema import ActionText
# from telnetlib import Telnet
from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment
from Path import Path
from ReplayBuffer import PrioritizedReplayBuffer
from Experience import Experience
from CentralAgent import CentralAgent
from Request import Request

from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

# import numpy as np
from copy import deepcopy
from os.path import join
from os import makedirs

# import pickle


class ValueFunction(ABC):

    def __init__(self, log_dir: str) -> None:
        super().__init__()

        log_dir = join(log_dir, type(self).__name__ + "/")
        makedirs(log_dir, exist_ok=True)

        self.summary_writer = SummaryWriter(log_dir)

    def add_to_logs(self, tag: str, value: float, step: int) -> None:
        self.summary_writer.add_scalar(tag, value, step)

    @abstractmethod
    def get_value(
        self, experience: List[Experience]
    ) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError()

    @abstractmethod
    def update(self, central_agent: CentralAgent):
        raise NotImplementedError()

    @abstractmethod
    def remember(self, experience: Experience):
        raise NotImplementedError()


class Q_Value_NN(nn.Module):
    """
    Value function approximator
    """

    def __init__(self, num_locs: int, max_cap: int) -> None:
        super().__init__()

        self.num_locs = num_locs
        self.max_cap = max_cap

        # num_embeddings 指定了嵌入层能够表示的离散值的数量
        # embedding_dim 表示嵌入的维度，即每个离散值将被映射到多少维的实数向量
        # padding_idx 是用于指定填充值的索引，当输入序列中出现填充值时，嵌入层会将其映射为全零向量
        self.location_embed = nn.Embedding(
            num_embeddings=num_locs + 1, embedding_dim=100, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=100 + 1,
            hidden_size=200,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.time_embed = nn.Linear(in_features=1, out_features=100)
        self.state_embed = nn.Sequential(
            nn.Linear(in_features=200 * 2 + 100 + 2, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=300),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(300, 1)

    def forward(self, data_inputs: List[Tuple[torch.Tensor]]) -> List[torch.Tensor]:

        # path_location_input (num_agents, max_cap*2+1)
        # delay_input (num_agents, max_cap*2+1)
        # current_time_input (num_agents,)
        # other_agents_input (num_agents,)
        # num_requests_input (num_agents,)

        outputs: List[torch.Tensor] = []

        for _, data in enumerate(data_inputs):
            (
                path_location_input,
                delay_input,
                current_time_input,
                other_agents_input,
                num_requests_input,
            ) = data

            path_location_embed = self.location_embed(
                path_location_input.long()
            )  # (num_actions*num_agents, max_cap*2+1, embedding_dim)
            delay_masked = torch.masked_fill(
                delay_input, delay_input == -1, 0
            )  # (num_actions*num_agents, max_cap*2+1)
            path_input = torch.concatenate(
                [path_location_embed, delay_masked.unsqueeze(-1)], dim=-1
            )  # (num_actions*num_agents, max_cap*2+1, embedding_dim+1)

            path_embed, _ = self.lstm(
                path_input
            )  # (num_actions*num_agents, max_cap*2+1, hidden_size*2)

            current_time_embed: torch.Tensor = self.time_embed(
                current_time_input.unsqueeze(-1)
            )  # (num_actions*num_agents, 100)
            current_time_embed = current_time_embed.unsqueeze(1).repeat(
                1, len(path_embed[1]), 1
            )  # (num_actions*num_agents, max_cap*2+1, 100)

            other_agents_input: torch.Tensor = (
                other_agents_input.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, len(path_embed[1]), 1)
            )
            # (num_actions*num_agents, max_cap*2+1, 1)

            num_requests_input: torch.Tensor = (
                num_requests_input.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, len(path_embed[1]), 1)
            )
            # (num_actions*num_agents, max_cap*2+1, 1)

            state_embed_input = torch.concatenate(
                [
                    path_embed,
                    current_time_embed,
                    other_agents_input,
                    num_requests_input,
                ],
                dim=-1,
            )
            # .view(len(path_embed[0]), len(path_embed[1]) * (len(path_embed[2]) + 102))
            # (num_actions*num_agents, max_cap*2+1, hidden_size*2+100+1+1)
            state_embed: torch.Tensor = self.state_embed(state_embed_input)
            # (num_actions*num_agents, max_cap*2+1, 300)

            outputs.append(torch.sum(self.output_layer(state_embed), dim=1))
            # (num_actions*num_agents, max_cap*2+1, 1)
            # (num_actions*num_agents, 1)

        return outputs


class DataBatch:
    """
    DataBatch for the Value Function
    """

    def __init__(
        self,
        experiences: List[Experience],
        is_current: bool,  # 是否处理当前时间步
        envt: Environment,
    ) -> None:

        self.envt = envt
        self.action_inputs_all_agents: List[Dict[str, List[Any]]] = []

        for experience in experiences:
            if is_current:
                if "current" not in experience.representation.keys():
                    experience.representation["current"] = self._format_input_batch(
                        [[agent] for agent in experience.agents],
                        experience.time,
                        experience.num_requests,
                    )
                input = deepcopy(experience.representation["current"])
            else:
                if "next" not in experience.representation.keys():
                    experience.representation["next"] = (
                        self._get_input_batch_next_state(experience)
                    )
                input = deepcopy(experience.representation["next"])

            self.action_inputs_all_agents.append(input)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor], List[int]]:
        """
        Getting formatted inputs for the state
        Args:
        ------
            index (int): Index of the batch

        Returns:
        ------
            Tuple[Tuple[torch.Tensor], List[int]]: Formatted inputs for the state
        """

        path_location_input: torch.Tensor
        delay_input: torch.Tensor
        current_time_input: torch.Tensor
        other_agents_input: torch.Tensor
        num_requests_input: torch.Tensor

        data_batch: List[Tuple[torch.Tensor]] = []
        shape_info_batch: List[List[int]] = []

        for action_input_all_agents in self.action_inputs_all_agents:
            for _, value in action_input_all_agents.items():
                shape_info: List[int] = []
                for list_elem in value:
                    shape_info.append(len(list_elem))

            shape_info_batch.append(shape_info)

            path_location_input = torch.stack(
                [
                    action
                    for agent in action_input_all_agents["path_location_input"]
                    for action in agent
                ]
            )
            # (num_actions*num_agents, max_cap*2+1)

            delay_input = torch.stack(
                [
                    action
                    for agent in action_input_all_agents["delay_input"]
                    for action in agent
                ]
            )
            # (num_actions*num_agents, max_cap*2+1)

            current_time_input = torch.tensor(
                [
                    action
                    for agent in action_input_all_agents["current_time_input"]
                    for action in agent
                ],
                dtype=torch.float32,
            )
            # (num_actions*num_agents,)

            other_agents_input = torch.tensor(
                [
                    action
                    for agent in action_input_all_agents["other_agents_input"]
                    for action in agent
                ],
                dtype=torch.float32,
            )
            # (num_actions*num_agents,)

            num_requests_input = torch.tensor(
                [
                    action
                    for agent in action_input_all_agents["num_requests_input"]
                    for action in agent
                ],
                dtype=torch.float32,
            )
            # (num_actions*num_agents,)

            data_batch.append(
                (
                    path_location_input,
                    delay_input,
                    current_time_input,
                    other_agents_input,
                    num_requests_input,
                )
            )

        return (data_batch[index], shape_info_batch[index])

    def _format_input(
        self,
        agent: LearningAgent,
        current_time: float,
        num_requests: float,
        num_other_agents: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
        """
        Getting formatted inputs for the state
        Args:
            agent (LearningAgent): Agent object
            current_time (float): Current time
            num_requests (float): Number of requests
            num_other_agents (float): Number of other agents

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, float, float]: Formatted inputs for the state
        """
        # Normalising Inputs
        current_time_input = (current_time - self.envt.START_EPOCH) / (
            self.envt.STOP_EPOCH - self.envt.START_EPOCH
        )
        num_requests_input = num_requests / self.envt.NUM_AGENTS
        num_other_agents_input = num_other_agents / self.envt.NUM_AGENTS

        # Getting path based inputs
        location_order: torch.Tensor = torch.zeros(
            size=(self.envt.MAX_CAPACITY * 2 + 1,), dtype=torch.float32
        )
        delay_order: torch.Tensor = (
            torch.zeros(size=(self.envt.MAX_CAPACITY * 2 + 1,), dtype=torch.float32) - 1
        )

        # Adding current location
        location_order[0] = agent.position.next_location + 1
        delay_order[0] = 1

        for idx, node in enumerate(agent.path.request_order):

            if idx >= 2 * self.envt.MAX_CAPACITY:
                break

            location, deadline = agent.path.get_info(node)
            visit_time = node.expected_visit_time

            location_order[idx + 1] = location + 1
            delay_order[idx + 1] = (deadline - visit_time) / Request.MAX_DROPOFF_DELAY

        return (
            location_order,
            delay_order,
            current_time_input,
            num_requests_input,
            num_other_agents_input,
        )

    def _format_input_batch(
        self,
        all_agents_post_actions: List[List[LearningAgent]],
        current_time: float,
        num_requests: int,
    ):  # Dict[str, List[List[Any]]]
        """
        Getting formatted inputs for the state
        Args:
        ------
            all_agents_post_actions (List[List[LearningAgent]]): List of agents post actions
            current_time (float): Current time
            num_requests (int): Number of requests

        Returns:
        ------
            Dict[str, List[List[Any]]]: Formatted inputs for the state

        """

        input: Dict[str, List[List[Any]]] = {
            "path_location_input": [],
            "delay_input": [],
            "current_time_input": [],
            "other_agents_input": [],
            "num_requests_input": [],
        }

        # Format all the other inputs
        for agent_post_actions in all_agents_post_actions:
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_agents_input = []

            # Get number of surrounding agents
            current_agent = agent_post_actions[0]
            # Assume first action is _null_ action
            num_other_agents = 0
            for other_agents_post_actions in all_agents_post_actions:
                other_agent = other_agents_post_actions[0]
                if (
                    self.envt.get_travel_time(
                        current_agent.position.next_location,
                        other_agent.position.next_location,
                    )
                    < Request.MAX_PICKUP_DELAY
                    or self.envt.get_travel_time(
                        other_agent.position.next_location,
                        current_agent.position.next_location,
                    )
                    < Request.MAX_PICKUP_DELAY
                ):
                    num_other_agents += 1

            for agent in agent_post_actions:
                # Get formatted output for the state
                (
                    location_order,
                    delay_order,
                    current_time_scaled,
                    num_requests_scaled,
                    num_other_agents_scaled,
                ) = self._format_input(
                    agent, current_time, num_requests, num_other_agents
                )

                current_time_input.append(current_time_scaled)
                num_requests_input.append(num_requests_scaled)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_agents_input.append(num_other_agents_scaled)

            input["current_time_input"].append(current_time_input)  # List[List[float]]
            input["num_requests_input"].append(num_requests_input)  # List[List[float]]
            input["delay_input"].append(delay_input)  # List[List[torch.Tensor]]
            input["path_location_input"].append(
                path_location_input
            )  # List[List[torch.Tensor]]
            input["other_agents_input"].append(other_agents_input)  # List[List[float]]

        return input

    def _get_input_batch_next_state(
        self, experience: Experience
    ):  # -> Dict[str, List[List[Any]]]:
        """
        Getting input for the next state

        Args:
            experience (Experience): Experience object

        Returns:
            Dict[str, List[List[Any]]]: Inputs for the next state
        """

        all_agents_post_actions: List[List[LearningAgent]] = []

        for agent, feasible_actions in zip(
            experience.agents, experience.feasible_actions_all_agents
        ):
            agents_post_actions: List[LearningAgent] = []
            for action in feasible_actions:
                # Moving agent according to feasible action
                agent_next_time = deepcopy(agent)
                assert action.new_path
                agent_next_time.path = deepcopy(action.new_path)
                self.envt.simulate_motion([agent_next_time], rebalance=False)

                agents_post_actions.append(agent_next_time)
            all_agents_post_actions.append(agents_post_actions)

        next_time = experience.time + self.envt.EPOCH_LENGTH

        # Return formatted inputs of these agents
        return self._format_input_batch(
            all_agents_post_actions, next_time, experience.num_requests
        )


class PathBasedNN(ValueFunction):

    def __init__(
        self,
        envt: Environment,
        load_model_loc: str,
        log_dir: str,
        GAMMA: float = -1,
        BATCH_SIZE_TRAIN: int = 8,
        BATCH_SIZE_PREDICT: int = 16,
        TARGET_UPDATE_TAU: float = 0.1,
    ) -> None:
        super().__init__(log_dir)

        self.envt = envt
        self.GAMMA = GAMMA if GAMMA != -1 else (1 - (0.1 * 60 / self.envt.EPOCH_LENGTH))
        self.BATCH_SIZE_TRAIN = BATCH_SIZE_TRAIN
        self.BATCH_SIZE_PREDICT = BATCH_SIZE_PREDICT
        self.TARGET_UPDATE_TAU = TARGET_UPDATE_TAU

        self._epoch_id = 0

        MIN_LEN_REPLAY_BUFFER = 1e6 / self.envt.NUM_AGENTS
        epochs_in_episode = (
            self.envt.STOP_EPOCH - self.envt.START_EPOCH
        ) / self.envt.EPOCH_LENGTH
        len_replay_buffer = max(MIN_LEN_REPLAY_BUFFER, epochs_in_episode)
        self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

        if not load_model_loc:
            self.model: nn.Module = Q_Value_NN(
                self.envt.NUM_LOCATIONS, self.envt.MAX_CAPACITY
            )
        else:
            self.model.load_state_dict(torch.load(load_model_loc))

        self.target_model: nn.Module = Q_Value_NN(
            self.envt.NUM_LOCATIONS, self.envt.MAX_CAPACITY
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.update_target_model = self._soft_update_fun()

    def _soft_update_fun(self) -> None:
        for target_param, source_param in zip(
            self.target_model.state_dict().items(),
            self.model.state_dict().items(),
        ):
            target_param = (
                target_param[0],
                self.TARGET_UPDATE_TAU * source_param[1]
                + (1.0 - self.TARGET_UPDATE_TAU) * target_param[1],
            )

    def remember(self, experience: Experience) -> None:
        """Remembers an experience for training."""
        self.replay_buffer.add(experience)

    def update(self, central_agent: CentralAgent, num_epoches: int = 3) -> None:
        """
        Updates the policy based on experiences in the replay buffer.
        The policy is updated using a Q-learning algorithm.
        The target policy is updated using a soft update function.

        Args:
        ------
            central_agent (CentralAgent): Central agent object
            num_epoches (int): Number of epochs to train the policy
        """
        # Check if replay buffer has enough samples for an update
        num_min_train_samples = int(5e5 / self.envt.NUM_AGENTS)
        if num_min_train_samples > len(self.replay_buffer):
            return

        # SAMPLE FROM REPLAY BUFFER
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # TODO: Implement Beta Scheduler
            beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
            experiences, weights, batch_idxes = self.replay_buffer.sample(
                self.BATCH_SIZE_TRAIN, beta
            )
        else:
            experiences = self.replay_buffer.sample(self.BATCH_SIZE_TRAIN)
            weights = None

        # ITERATIVELY UPDATE POLICY BASED ON SAMPLE
        for experience_idx, experience in enumerate(experiences):
            if weights is not None:
                weights = torch.tensor([weights[experience_idx]]).repeat(
                    self.envt.NUM_AGENTS
                )  # (num_agents,)

            scored_actions_all_agents = self.get_value(
                [experience], network=self.target_model
            )

            value_next_state: List[float] = []

            final_actions = central_agent.choose_actions(
                scored_actions_all_agents, is_training=False
            )

            data_input_all_actions, data_shape_info_all_actions = DataBatch(
                [experience], is_current=True, envt=self.envt
            )[:]

            feasible_actions: List[List[Action]] = (
                experience.feasible_actions_all_agents
            )
            data_input: List[torch.Tensor] = []

            cumulative_num_actions: int = 0
            for id1, (action, score) in enumerate(final_actions):
                value_next_state.append(score)
                action_index: int = feasible_actions[id1].index(action)

                for id2, tensor in enumerate(data_input_all_actions[0]):
                    data_input[id2] = torch.concatenate(
                        [
                            data_input[id2],
                            tensor[cumulative_num_actions + action_index],
                        ],
                        dim=-1,
                    )
                cumulative_num_actions += data_shape_info_all_actions[0][id1]

            data_input: List[Tuple[torch.Tensor]] = [tuple(data_input)]

            supervised_targets = torch.tensor(value_next_state).view(-1, 1)

            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            loss_fun = nn.functional.mse_loss

            self.model.train()
            for _ in range(num_epoches):
                optimizer.zero_grad()
                predicted_targets: torch.Tensor = (self.model(data_input))[0]
                loss = loss_fun(predicted_targets, supervised_targets)
                weighted_loss = loss * weights.unsqueeze(1)
                mean_weighted_loss = torch.mean(weighted_loss)
                mean_weighted_loss.backward()

                optimizer.step()

    def _reconstruct_NN_output(
        self, NN_outputs: List[torch.Tensor], shape_infos: List[List[int]]
    ) -> List[List[float]]:
        """
        Reconstructs the NN output as a list of lists of floats.
        Each list represents the output of a single agent.
        The list of lists represents the output of all agents.

        Args:
            NN_outputs (List[torch.Tensor]): List of NN outputs.
            shape_infos (List[List[int]]): List of shape infos.

        Returns:
            List[List[float]]: Reconstructed list of lists of floats.
        """

        output_as_list: List[List[float]] = []
        for tensor, shape_info in zip(NN_outputs, shape_infos):
            # tensor: (num_actions*num_agents, 1)
            # shape_info: [num_actions_1, num_actions_2,...,num_actions_(num_agents)]
            value_list: List[float] = tensor.squeeze().tolist()
            cumulative_num_actions: int = 0
            for num_actions in shape_info:
                value_per_agent: List[float] = value_list[
                    cumulative_num_actions : cumulative_num_actions + num_actions
                ]
                cumulative_num_actions += num_actions
                output_as_list.append(value_per_agent)

        return output_as_list

    def get_value(
        self, experiences: List[Experience], network: nn.Module = None
    ) -> List[List[Tuple[Action, float]]]:
        """
        Returns a list of lists of tuples of actions and values.
        Each tuple represents the action and the value of the action.
        Each list represents the actions and values of a single agent.
        The list of lists represents the actions and values of all agents.

        Args:
        ------
            experiences (List[Experience]): List of experiences.
            network (nn.Module, optional): Network to use for computing values. Defaults to None.

        Returns:
        ------
            List[List[Tuple[Action, float]]]: List of lists of tuples of actions and values.
        """

        data_inputs, data_shape_info = DataBatch(
            experiences, is_current=False, envt=self.envt
        )[:]
        # Tuple[List[Tuple[torch.tensor]], List[List[int]]]
        # data_inputs: List[Tuple[torch.tensor]]
        # data_shape_info: List[List[int]]

        self.model.eval()

        if network is None:
            expected_future_values_all_agents: List[torch.Tensor] = self.model(
                data_inputs
            )
        else:
            expected_future_values_all_agents: List[torch.Tensor] = network(data_inputs)

        expected_future_values_all_agents: List[List[float]] = (
            self._reconstruct_NN_output(
                expected_future_values_all_agents, data_shape_info
            )
        )

        def get_score(action: Action, value: float):
            return self.envt.get_reward(action) + self.GAMMA * value

        feasible_actions_all_agents: List[List[Action]] = [
            feasible_actions
            for experience in experiences
            for feasible_actions in experience.feasible_actions_all_agents
        ]

        scored_actions_all_agents: List[List[Tuple[Action, float]]] = []

        for expected_future_values, feasible_actions in zip(
            expected_future_values_all_agents, feasible_actions_all_agents
        ):
            scored_actions = [
                (action, get_score(action, value))
                for value, action in zip(expected_future_values, feasible_actions)
            ]
            scored_actions_all_agents.append(scored_actions)

        return scored_actions_all_agents
