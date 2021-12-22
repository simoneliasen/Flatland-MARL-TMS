import os
from torch.distributions.categorical import Categorical
import torch as torch
import torch.nn as nn
import torch.optim as optim


class ActorCriticModel(nn.Module):
    """
    An MLP network for both the actor and critic. 
    
    Parameters:
    in_size (int): Size of the input layer
    out_size (int): Size of the output layer
    h_size (int): Size of the hidden layer(s)
    h_num (int): Number of hidden layers
    h_act_fnc (string): Activation function in input and hidden layers.

    Return:
    policy (nn.Tensor): A probability distribution of the actions 
    """

    def __init__(self, in_size, out_size, device, h_size, h_num, h_act_fnc):
        super(ActorCriticModel, self).__init__()
        self.device = device

        # Activation function
        if(h_act_fnc == "ReLU"):
            activation = nn.ReLU()
        if(h_act_fnc == "Tanh"):
            activation = nn.Tanh()
        if(h_act_fnc == "Sigmoid"):
            activation = nn.Sigmoid()
        if(h_act_fnc == "LeakyReLU"):
            activation = nn.LeakyReLU()

        # ----- Actor Model ----- #
        actor_layers = nn.ModuleList()
        
        # Input layer
        actor_layers.append(nn.Linear(in_size, h_size))
        actor_layers.append(activation)

        # Hidden layer(s)
        for i in range (h_num):
            actor_layers.append(nn.Linear(h_size, h_size))
            actor_layers.append(activation)
        
        # Output layer
        actor_layers.append(nn.Linear(h_size, out_size))
        actor_layers.append(nn.Softmax(dim=-1))

        # Define model in Pytorch
        self.actor = nn.Sequential(*actor_layers).to(self.device)       
        
        # ----- Critic Model ----- #
        critic_layers = nn.ModuleList()
        
        # Input layer
        critic_layers.append(nn.Linear(in_size, h_size)) 
        critic_layers.append(activation)

        # Hidden Layer(s)
        for i in range (h_num):
            critic_layers.append(nn.Linear(h_size, h_size))
            critic_layers.append(activation)
        
        # Output layer
        critic_layers.append(nn.Linear(h_size, 1)) 
        critic_layers.append(nn.Identity())

        # Define network structure in Pytorch
        self.critic = nn.Sequential(*critic_layers).to(self.device)


    def forward(self, x):
        raise NotImplementedError

    def get_actor_dist(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        return dist

    def get_critic_state_value(self, state):
        state_value = self.critic(state)
        return state_value

    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_value = self.critic(states)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.actor.state_dict(), filename + ".actor")
        torch.save(self.critic.state_dict(), filename + ".value")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.actor = self._load(self.actor, filename + ".actor")
        self.critic = self._load(self.critic, filename + ".value")