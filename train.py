import os
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from pprint import pprint
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
#Flatland Imports
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.utils.rendertools import RenderTool
#Tools
from tools.normalize_observation import normalize_observation 
from tools.skip_no_choice_cells import RailEnvWrapper, SkipNoChoiceCellsWrapper
from tools.agent_action_config import get_flatland_full_action_size, get_action_size, map_actions, map_action, set_action_size_reduced, set_action_size_full, map_action_policy
from tools.env_configuration import create_rail_env
from tools.eval_policy import eval_policy
from tools.deadlock_check import Deadlock
from tools.timer import Timer
#Observations
from flatland.envs.observations import TreeObsForRailEnv as TreeObs
from observations.NewTreeObs import TreeObsForRailEnv as NewTreeObs
#Models 
from models.PPO.ppo_agent import PPOPolicy as PPO
from models.DDQN.ddqn_policy import DDDQNPolicy as DDDQN

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


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

def train_agent(train_params, train_env_params, eval_env_params, obs_params, env_params, other_env_params, ppo_parameters = None):

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    #Environment parameters
    n_agents = train_env_params.n_agents

    if train_params.curriculum == "progressive":
        n_agents = 5

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
  
    def check_is_observation_valid(observation):
        return observation

    #Decide between what normalization method to use.
    def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
        if training_params.observation == "NewTreeObs":
            return normalize_observation(observation, tree_depth, training_params.observation)
        elif training_params.observation == "TreeObs":
            return normalize_observation(observation, tree_depth, training_params.observation)
        else:
            return print("error")

    #New or old TreeObs
    if training_params.observation == "NewTreeObs":
        tree_observation = NewTreeObs(max_depth=observation_tree_depth, predictor=predictor)
        print("\n>> NewTreeObs")
    elif training_params.observation == "TreeObs":
        tree_observation = TreeObs(max_depth=observation_tree_depth, predictor=predictor)
        print("\n>> TreeObs")
    else:
        print("No observation was set")

    tree_observation.check_is_observation_valid = check_is_observation_valid
    tree_observation.get_normalized_observation = get_normalized_observation
   
    # Setup some environment for training
    train_env = create_rail_env(train_env_params, tree_observation)

    if training_params.skip_no_choice:
        train_env.reset(regenerate_schedule=True, regenerate_rail=True)
        train_env = RailEnvWrapper(train_env)
        SkipNoChoiceCellsWrapper(train_env, accumulate_skipped_rewards=False, discounting=0.0)
        
    # Setup some environment for evaluation
    eval_env = create_rail_env(eval_env_params, tree_observation)

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    # Smoothed values used as target for hyperparameter tuning
    smoothed_eval_normalized_score = -1.0
    smoothed_eval_completion = 0.0

    scores_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
    completion_window = deque(maxlen=checkpoint_interval)

    #Reduced or full action_size
    if train_params.action_size == "reduced":
        set_action_size_reduced()
    else:
        set_action_size_full()

    #Set PPO as policy
    if train_params.policy == "PPO":
        policy = PPO(state_size, get_action_size(), use_replay_buffer=True, in_parameters=train_params)
    elif train_params.policy == "DDDQN":
        policy = DDDQN(state_size, get_action_size(), train_params)
    else:
        print("no policy was set")

    #Make sure that at least one policy is set
    if policy is None:
        policy = PPO(state_size, get_action_size(), use_replay_buffer=True, in_parameters=train_params)

    if train_params.sweep == True:
        policy = PPO(state_size, get_action_size(), use_replay_buffer=True, sweep = True, sweep_parameters = ppo_parameters)

    # Load existing policy
    if train_params.load_policy != "":
        policy.load(train_params.load_policy)

    print(">> Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    # TensorBoard writer
    writer = SummaryWriter(comment="_" + train_params.policy + "_" + train_params.action_size)

    #General level counter for env_creation
    level = 0
    
    #Random Curriculum setup
    num_epochs = 10000
    level_sequence = []
    #create level entries for each epoch
    for i in range(5): # 
        level_sequence += [i] * num_epochs
    #Shuffle randomly
    random.shuffle(level_sequence)
    list_iter = 0

    training_timer = Timer()
    training_timer.start()

    for episode_idx in range(n_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        if training_params.sweep == True and episode_idx == 2500:
            break;
        # Reset environment
        reset_timer.start()

        # Infinite epochs (manually spot convergence)
        if train_params.curriculum == "sequential": 
            if level == training_params.training_subset: #Runs until specified subset of trainingset (5) is reached
                level = 0
            train_env_params.n_agents = env_params[level]["n_agents"]
            number_of_agents = train_env_params.n_agents
            train_env_params.x_dim = env_params[level]["x_dim"]
            train_env_params.y_dim = env_params[level]["y_dim"]
            train_env_params.n_cities = env_params[level]["n_cities"]
            level+=1

        elif train_params.curriculum == "incremental":  
            if level == training_params.training_subset: #Runs until specified subset of trainingset is reached
                break
            train_env_params.n_agents = env_params[level]["n_agents"]
            number_of_agents = train_env_params.n_agents
            train_env_params.x_dim = env_params[level]["x_dim"]
            train_env_params.y_dim = env_params[level]["y_dim"]
            train_env_params.n_cities = env_params[level]["n_cities"]
        
        # We should probably just run this for same epochs as others and see if efficient
        elif train_params.curriculum == "random": #we should probably just run this for same epochs as others and see if efficient
            if list_iter <= len(level_sequence):
                train_env_params.n_agents = env_params[level_sequence[list_iter]]["n_agents"]
                number_of_agents = train_env_params.n_agents
                train_env_params.x_dim = env_params[level_sequence[list_iter]]["x_dim"]
                train_env_params.y_dim = env_params[level_sequence[list_iter]]["y_dim"]
                train_env_params.n_cities = env_params[level_sequence[list_iter]]["n_cities"]
                list_iter += 1
            else:
                print("Broke because of error in code")
                break;

        #Progressively works up to full sequential run     
        elif train_params.curriculum == "progressive":
            number_of_agents = int(min(n_agents, 1 + np.floor(episode_idx / 200)))
            env_index = episode_idx % number_of_agents + 1
            train_env_params.n_agents = env_params[env_index - 1]["n_agents"]
            number_of_agents = train_env_params.n_agents
           
        #Fixed level
        elif train_params.curriculum == "fixed":
            number_of_agents = train_env_params.n_agents 
           # if early_stop:
               # print("Early stopping triggered")

        else:
            print("You didn't select curriculum")

        n_cities = train_env_params.n_cities
  
        action_count = [0] * get_flatland_full_action_size()
        action_dict = dict()
        agent_obs = [None] * number_of_agents
        agent_prev_obs = [None] * number_of_agents
        agent_prev_action = [2] * number_of_agents
        update_values = [False] * number_of_agents

        train_env = create_rail_env(train_env_params, tree_observation)
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True) #NEcessary?
        policy.reset(train_env)
        reset_timer.end()

        if train_params.render:
            # Setup renderer
            env_renderer = RenderTool(train_env, gl="PGL")
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent_handle in train_env.get_agent_handles():
            if tree_observation.check_is_observation_valid(obs[agent_handle]):
                agent_obs[agent_handle] = tree_observation.get_normalized_observation(obs[agent_handle], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()
        
        max_steps = train_env._max_episode_steps

        # Run episode
        policy.start_episode(train=True)
        for step in range(max_steps):
            inference_timer.start()
            policy.start_step(train=True)

            for agent_handle in train_env.get_agent_handles():
                #print(train_env.agents[agent_handle].state)
                agent = train_env.agents[agent_handle]
                if info['action_required'][agent_handle]:
                    update_values[agent_handle] = True
                    action = policy.act(agent_handle, agent_obs[agent_handle])
                    action_count[map_action(action)] += 1
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent_handle] = False
                    action = 0
                action_dict.update({agent_handle: action})
                actions_taken.append(map_action(action))
            policy.end_step(train=True)
            inference_timer.end()

            # Environment step
            step_timer.start()
            if train_params.action_size == "full":
                next_obs, all_rewards, done, info = train_env.step(action_dict) #Saves an iteration over 5 RailEnvActions
            elif train_params.action_size == "reduced":
                next_obs, all_rewards, done, info = train_env.step(map_actions(action_dict)) 
            step_timer.end()

            # Render an episode at some interval
            if train_params.render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent_handle in train_env.get_agent_handles():
                if update_values[agent_handle] or done['__all__']:

                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent_handle,
                                agent_prev_obs[agent_handle],
                                map_action_policy(agent_prev_action[agent_handle]), 
                                all_rewards[agent_handle],
                                agent_obs[agent_handle],
                                done[agent_handle])
                    learn_timer.end()

                    agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()
                    agent_prev_action[agent_handle] = action_dict[agent_handle]

                # Preprocess the new observations
                if tree_observation.check_is_observation_valid(next_obs[agent_handle]):
                    preproc_timer.start()
                    agent_obs[agent_handle] = tree_observation.get_normalized_observation(next_obs[agent_handle],
                                                                                          observation_tree_depth,
                                                                                          observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent_handle]

            nb_steps = step

            if done['__all__']:
                break

        policy.end_episode(train=True)

        #Deadlock data
        deadlockData = Deadlock.deadlock_data(train_env)
        unfinished_agents_num = deadlockData[0]
        unfinished_not_deadlock = deadlockData[1]
        deadlocks = deadlockData[2]

        # Collect information about training
        tasks_finished = sum(train_env.agents[a].state == TrainState.DONE for a in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents()) ##########################
        normalized_score = score / (max_steps * train_env.get_num_agents())

        #Try/Except: as error arises especially in small levels when a lot of actions are 0 (dividing by zero)
        try: 
            action_probs = action_count / max(1, np.sum(action_count))
        except:
            print("Error loading action_probs")

            action_probs = [0, 0, 0, 0, 0]

        scores_window.append(normalized_score)
        completion_window.append(completion) ###########################
        smoothed_normalized_score = np.mean(scores_window)
        smoothed_completion = np.mean(completion_window) ############################

        if train_params.render:
            env_renderer.close_window()

        # Print logs
        if episode_idx % checkpoint_interval == 0 and episode_idx > 0:
            policy.save( 'models/PPO/checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

            # Reset action count: full even if reduced actions, to debug and analyze
            action_count = [0] * get_flatland_full_action_size()

        print(
            '\r🚂 Episode {}'
            '\t 🚉 nAgents {:2}/{:2}'
            ' 🏆 Score: {:7.3f}'
            ' Avg: {:7.3f}'
            '\t 💯 Done: {:6.2f}%'
            ' Avg: {:6.2f}%'
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                train_env_params.n_agents, number_of_agents,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0 and episode_idx != 0:
            if training_params.evaluate:
                deadlock_data, scores, completions, nb_steps_eval = eval_policy(eval_env, policy, tree_observation, train_params, obs_params, other_env_params, training_params.observation)
                unfinished_agents_num_eval, unfinished_not_deadlock_eval, deadlocks_eval, is_deadlock_eval = deadlock_data 

                # Save validation logs to tensorboard
                smoothing = 0.9
                writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
                writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
                writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx) #Normalized score
                writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
                writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
                writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
                writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
                writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx) #Completions
                writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
                writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
                writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
                writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
                writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
                writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
                writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)
                writer.add_scalar('evaluation/deadlocks_num', np.mean(deadlocks_eval), episode_idx) #Deadlocks
                writer.add_scalar('evaluation/unfinished_agents_num', np.mean(unfinished_agents_num_eval), episode_idx) #Total unfinished agents
                writer.add_scalar('evaluation/unfinished_not_deadlock', np.mean(unfinished_not_deadlock_eval), episode_idx) #Unifnished - not deadlocked
                smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
                smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
                writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx) 
                writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx) 

        # Save training logs to tensorboard
        if train_params.sweep == True:
            wandb.log({"scores": normalized_score, "smoothed_score": smoothed_normalized_score, "completion": np.mean(completion), "smoothed_completion": np.mean(smoothed_completion),"total_time": training_timer.get_current()})
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/score_mean", np.mean(normalized_score), episode_idx) #Normalized score
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx) #Completions
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_scalar("training/n_agents", train_env_params.n_agents, episode_idx)
        writer.add_scalar('training/deadlocks_num', np.mean(deadlocks), episode_idx) #Deadlocks
        writer.add_scalar('training/unfinished_agents_num', np.mean(unfinished_agents_num), episode_idx) #Total unfinished agents
        writer.add_scalar('training/unfinished_not_deadlock', np.mean(unfinished_not_deadlock), episode_idx)#Unifnished - not deadlocked
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx) 
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx) 
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx) 
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx) 
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx) 
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx) 
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)
        writer.flush()

#Format action distribution for commandoprompt
def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer



if __name__ == "__main__":

    sweep_config = {
    'method': 'random', #grid, random, bayesian
    'metric': {
    'name': 'smoothed_score',
    'goal': 'maximize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [64, 128, 256] 
        },
        'epochs': {
            'values': [3, 5, 10] #reduced
        },
        'discount_gamma': {
            'values': [0.95, 0.97, 0.99]
        },
        'gae_lambda': { 
            'values': [0.95]
        },
        'lr': { 
            'values': [0.000005, 0.00005, 0.0005]
        },
        'clip': {
            'values': [0.1, 0.2] #reduced
        },
        'num_hid_layers': { 
            'values': [2]
        },
        'hidden_size': { #Could implement different sizes for different layers
            'values': [64, 128, 256]
        },
        'value_coefficient': {
            'values': [0.5, 1]
        },
        'entropy_coefficient': {
            'values': [0, 0.01]
        },
        'activation_function': { #Sigmoid removed to test if clean run is possible w. default
            'values': ["Tanh", "ReLU"]  
        },
        'buffer_size': { 
            'values': [200, 250, 300]
        }
    }
}

    parser = ArgumentParser()
    #If fixed env, what environment to train on
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=0, type=int) 
    
    # PPO-related parameters
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(51200), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--batch_size", help="minibatch size", default=256, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--gae_lambda", help="GAE lambda factor", default=0.95, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.00005, type=float)
    parser.add_argument("--h_act_fnc", help="What activation function to use", default="ReLU", type=str)
    parser.add_argument("--hidden_size", help="size of the hidden layers", default=128, type=int)
    parser.add_argument("--h_num", help="number of hidden layers", default=2, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=10, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=12, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", action='store_true')
    parser.add_argument("--load_policy", help="policy filename (reference) to load", default="", type=str) #python train.py --load_policy models\PPO\checkpoints\211207145933-1400.pth
    parser.add_argument("--max_depth", help="max depth", default=2, type=int)
    parser.add_argument("--policy", help="policy name [PPO, DDDQN]", default="PPO")
    parser.add_argument("--action_size", help="define the action size [reduced,full]", default="full", type=str)
    parser.add_argument("--tau", help="soft update of target parameters", default=0.5e-3, type=float) # for DDQN




    #Additional parameters
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=1000000000, type=int) #Set ridicously high so it doesn't accidently
    parser.add_argument("--skip_no_choice", help="skip no-choice cells", default=True, type=bool)
    parser.add_argument("--sweep", help="run sweep on hyperparameters for tuning", default=False, type=bool)
    parser.add_argument("--evaluate", help="run evaluation every 100'th episode", default=True, type=bool)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=200, type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=5, type=int)
    parser.add_argument("--observation", help="define the observation used [NewTreeObs,TreeObs]", default="NewTreeObs", type=str)
    parser.add_argument("--curriculum", help="define curriculum used in training [sequential, incremental, rand, progressive, fixed]", default="progressive", type=str)
    parser.add_argument("--training_set", help="What training set to use [new, old]", default="new", type=str)
    parser.add_argument("--validate_set", help="What validation set to use [new, old]", default="old", type=str)
    parser.add_argument("--training_subset", help="subset of test-set for training", default=5, type=int) #How many environments to train on
    parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int) #Not really used
    training_params = parser.parse_args()

    if training_params.sweep == True:
        sweep_id = wandb.sweep(sweep_config, entity="simoneliasen", project="sweeps-tutorial")

    old_env_params = [ 
        {   #Test_00
            "n_agents": 7, 
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_01
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_02
            "n_agents": 20,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_03
            "n_agents": 50,
            "x_dim": 30,
            "y_dim": 35,
            "n_cities": 3,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_04
            "n_agents": 80,
            "x_dim": 35,
            "y_dim": 30,
            "n_cities": 5,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        }
    ]

    new_env_params = [ 
        {   #Test_00
            "n_agents": 2,  
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_01
            "n_agents": 4,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_02
            "n_agents": 8,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_03
            "n_agents": 12,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {   #Test_04
            "n_agents": 16,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "n_envs_run": 10,
            "malfunction_rate": 1 / 200,
            "seed": 0
        }
    ]

    obs_params = {
        "observation_tree_depth": training_params.max_depth,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
    }

    if training_params.training_set == "old":
        training_env_params = old_env_params[training_params.training_env_config]
        env_params = old_env_params
        other_env_params = new_env_params
    elif training_params.training_set == "new":
        training_env_params = new_env_params[training_params.training_env_config]
        env_params = new_env_params
        other_env_params = old_env_params
        print("new training_env_params param")
    else:
        print("no training-set was selected")

    if training_params.validate_set == "old":
        evaluation_env_params = old_env_params[training_params.evaluation_env_config]
    elif training_params.validate_set == "new":
        evaluation_env_params = new_env_params[training_params.evaluation_env_config]
        
    else: 
        print("no validation-set was selected")

    print("\n>> Training parameters:")
    pprint(vars(training_params))
    print("\n>> Observation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)

    if training_params.sweep == True:
        count = 1000000
        print("\n>>Running Sweep " + str(count) + " times")
        wandb.agent("wl76fb2o", function=sweep, count = count)

    else:
        train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params), Namespace(**obs_params), env_params , other_env_params)
