
import wandb
from train import train_agent, sweep_config, training_params, training_env_params, evaluation_env_params, obs_params, env_params, other_env_params
from argparse import Namespace

def sweep():
    wandb.init(config=sweep_config)
    gamma = wandb.config['discount_gamma']
    gae_lambda = wandb.config['gae_lambda']
    epsilon = wandb.config['clip']
    c1 = wandb.config['value_coefficient']
    c2 = wandb.config['entropy_coefficient']
    batch_size = wandb.config['batch_size']
    hidden_size = wandb.config['hidden_size'] #Not different hid-size for each layer
    h_num = wandb.config['num_hid_layers']
    act_fnc = wandb.config['activation_function'] 
    n_epochs = wandb.config['epochs']
    lr = wandb.config['lr'] #try same for both
    buffer_size = batch_size * wandb.config['buffer_size']
    ppo_parameters = gamma, gae_lambda, epsilon, c1, c2, batch_size, hidden_size, h_num, act_fnc, n_epochs, lr,  buffer_size
    train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params), Namespace(**obs_params), env_params , other_env_params, ppo_parameters)
