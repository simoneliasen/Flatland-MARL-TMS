import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
from models.PPO.policy import LearningPolicy
from models.PPO.replay_buffer import ReplayBuffer
from models.PPO.model import ActorCriticModel

# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})


class PPOPolicy(LearningPolicy):
    def __init__(self, state_size, action_size, use_replay_buffer=False, in_parameters=None, sweep = False, sweep_parameters = None):
        print(">> PPOPolicy")
        super(PPOPolicy, self).__init__()
        # parameters
        self.ppo_parameters = in_parameters
        if self.ppo_parameters is not None:
            self.h_size = self.ppo_parameters.hidden_size
            self.h_num = self.ppo_parameters.h_num
            self.h_act_fnc = self.ppo_parameters.h_act_fnc
            self.buffer_size = self.ppo_parameters.buffer_size
            self.batch_size = self.ppo_parameters.batch_size
            self.learning_rate = self.ppo_parameters.learning_rate
            self.gamma = self.ppo_parameters.gamma
            self.gae_lambda = self.ppo_parameters.gae_lambda
           
            # Device
            if self.ppo_parameters.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(">> Using GPU")
            else:
                self.device = torch.device("cpu")
                print(">> Using CPU")
        else:
            self.h_size = 128
            self.h_num = 2
            self.h_act_fnc = "ReLU"
            self.learning_rate = 5e-05
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.buffer_size = 51200
            self.batch_size = 256
            self.device = torch.device("cpu")

        self.surrogate_eps_clip = 0.2
        self.K_epoch = 5 
        self.weight_loss = 0.5 #value coefficient
        self.weight_entropy = 0 #entropy coefficient

        if sweep:
            self.gamma = sweep_parameters[0]
            self.gae_lambda = sweep_parameters[1]
            self.surrogate_eps_clip = sweep_parameters[2]
            self.weight_loss = sweep_parameters[3]
            self.weight_entropy = sweep_parameters[4]
            self.batch_size = sweep_parameters[5]
            self.h_size = sweep_parameters[6]
            self.h_num = sweep_parameters[7]
            self.h_act_fnc = sweep_parameters[8]
            self.K_epoch = sweep_parameters[9]
            self.learning_rate = sweep_parameters[10]
            self.buffer_size = sweep_parameters[11]
            print(">> SweepParameters",self.gamma, self.gae_lambda, self.surrogate_eps_clip, self.weight_loss, self.weight_entropy, self.batch_size, self.h_size, self.h_num, self.h_act_fnc, self.K_epoch, self.learning_rate, self.buffer_size)

        self.buffer_min_size = 0
        self.use_replay_buffer = use_replay_buffer
        self.current_episode_memory = EpisodeBuffers()
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.actor_critic_model = ActorCriticModel(state_size, action_size, self.device, h_size=self.h_size, h_num=self.h_num, h_act_fnc=self.h_act_fnc)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()  # nn.SmoothL1Loss()

    def reset(self, env):
        pass

    def act(self, handle, state):
        # sample a action to take
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor_critic_model.get_actor_dist(torch_state)
        action = dist.sample()
        return action.item()

    def step(self, handle, state, action, reward, state_next, done):
        # record transitions ([state] -> [action] -> [reward, state_next, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        torch_state_next = torch.tensor(state_next, dtype=torch.float).to(self.device)
        
        # evaluate actor
        dist = self.actor_critic_model.get_actor_dist(torch_state)
        action_logprobs = dist.log_prob(torch_action)
        state_value = self.actor_critic_model.get_critic_state_value(torch_state)
        state_value_next = self.actor_critic_model.get_critic_state_value(torch_state_next)
        transition = (state, action, reward, state_next, done, state_value.item(),state_value_next.item(), action_logprobs.item())
        self.current_episode_memory.push_transition(handle, transition)

    def _push_transitions_to_replay_buffer(self,
                                           state_list,
                                           action_list,
                                           reward_list,
                                           state_next_list,
                                           done_list,
                                           advantage_list,
                                           prob_a_list):
        for idx in range(len(reward_list)):
            state_i = state_list[idx]
            action_i = action_list[idx]
            reward_i = reward_list[idx]
            state_next_i = state_next_list[idx]
            done_i = done_list[idx]
            advantage_i = advantage_list[idx]
            prob_action_i = prob_a_list[idx]
            self.memory.add(state_i, action_i, reward_i, state_next_i, done_i, advantage_i, prob_action_i)

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        # build empty lists(arrays)
        state_list, action_list, return_list, state_next_list, done_list, prob_a_list, advantage_list = [], [], [], [], [], [], []

        # Initiate variables
        discount = 1
        advantage = 0
        discounted_reward = 0
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, done_i, state_value_i, state_value_next_i, prob_action_i = transition

            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

            if done_i:
                discounted_reward = 0
                advantage = 0
                done_list.insert(0, 1)
            else:
                done_list.insert(0, 0)

            # GAE Estimation
            advantage += discount*(reward_i + self.gamma*state_value_next_i*(1-int(done_i)) - state_value_i)
            discount *= self.gamma*self.gae_lambda
            advantage_list.insert(0, advantage)
         
            # Returns: Target values for state value function
            discounted_reward = reward_i + self.gamma * (1-int(done_i)) * discounted_reward
            return_list.insert(0,  discounted_reward)

        if self.use_replay_buffer:
            self._push_transitions_to_replay_buffer(state_list, action_list,
                                                    return_list, state_next_list,
                                                    done_list, advantage_list, prob_a_list)

        # convert data to torch tensors
        states, actions, returns, states_next, dones, advantages, prob_actions = \
            torch.tensor(np.array(state_list), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(action_list)).to(self.device), \
            torch.tensor(np.array(return_list), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(state_next_list), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(done_list), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(prob_a_list)).to(self.device)

        return states, actions, returns, states_next, dones, advantages, prob_actions

    def _get_transitions_from_replay_buffer(self, states, actions, returns, states_next, dones, advantages, probs_action):
        if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
            states, actions, returns, states_next, dones, advantages, probs_action = self.memory.sample()
            actions = torch.squeeze(actions)
            returns = torch.squeeze(returns)
            states_next = torch.squeeze(states_next)
            dones = torch.squeeze(dones)
            advantages = torch.squeeze(advantages)
            probs_action = torch.squeeze(probs_action)
        return states, actions, returns, states_next, dones, advantages, probs_action

    def train_net(self):
        # All agents have to propagate their experiences made during past episode
        for handle in range(len(self.current_episode_memory)):
            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, returns, states_next, dones, advantages, probs_action = self._convert_transitions_to_torch_tensors(agent_episode_history)

                # Optimize policy for K epochs:
                for k_loop in range(int(self.K_epoch)):

                    if self.use_replay_buffer:
                        states, actions, returns, states_next, dones, advantages, probs_action = \
                            self._get_transitions_from_replay_buffer(
                                states, actions, returns, states_next, dones, advantages, probs_action
                            )

                    # Evaluating actions (actor) and values (critic)
                    logprobs, state_values, dist_entropy = self.actor_critic_model.evaluate(states, actions)

                    # Finding the ratios (pi_thetas / pi_thetas_replayed):
                    ratios = torch.exp(logprobs - probs_action.detach())

                    # Finding Surrogate Loos
                    #advantages = rewards - critic_values.detach()

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1. - self.surrogate_eps_clip, 1. + self.surrogate_eps_clip) * advantages

                    # The loss function is used to estimate the gardient and use the entropy function based
                    # heuristic to penalize the gradient function when the policy becomes deterministic this would let
                    # the gradient becomes very flat and so the gradient is no longer useful.
                    loss = \
                        -torch.min(surr1, surr2) \
                        + self.weight_loss * self.loss_function(state_values, returns) \
                        - self.weight_entropy * dist_entropy

                    # Make a gradient step
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                    # Transfer the current loss to the agents loss (information) for debug purpose only
                    self.loss = loss.mean().detach().cpu().numpy()

        # Reset all collect transition data
        self.current_episode_memory.reset()

    def end_episode(self, train):
        if train:
            self.train_net()

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        self.actor_critic_model.save(filename)
        torch.save(self.optimizer.state_dict(), filename + ".optimizer")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        else:
            print(" >> file not found!")
        return obj

    def load(self, filename):
        print("load policy from file", filename)
        self.actor_critic_model.load(filename)
        print("load optimizer from file", filename)
        self.optimizer = self._load(self.optimizer, filename + ".optimizer")

    def clone(self):
        policy = PPOPolicy(self.state_size, self.action_size)
        policy.actor_critic_model = copy.deepcopy(self.actor_critic_model)
        policy.optimizer = copy.deepcopy(self.optimizer)
        return self
