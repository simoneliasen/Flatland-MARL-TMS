import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import TrainState
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.evaluators.client import FlatlandRemoteClient, TimeoutException

from models.DDQN.ddqn_policy import DDDQNPolicy as DDDQN
from models.PPO.ppo_agent import PPOPolicy as PPO

from torch.utils.tensorboard import SummaryWriter

from tools.skip_no_choice_cells import RailEnvWrapper, SkipNoChoiceCellsWrapper
from observations.NewTreeObs import TreeObsForRailEnv as NewTreeObs
from tools.agent_action_config import get_action_size, map_actions, set_action_size_reduced, set_action_size_full
from tools.deadlock_check import Deadlock
from tools.normalize_observation import normalize_observation
import os

local_submit = True
action_mask = False
frame_skip = True
VERBOSE = True # Print per-step logs 

obs_type = "NewTreeObs" #or 
load_policy = "PPO"
checkpoint = "./models/PPO/checkpoints/ancient/211218214902-10000.pth"

if local_submit:
    os.environ['AICROWD_TESTS_FOLDER'] = "C:\\Users\\simon\\Desktop\\flatland-starter-kit\\debug-environments"

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

run_name = checkpoint[25:]

#C:\Users\simon\Desktop\flatland-starter-kit\models\PPO\checkpoints\Simon_incremental_7agents_oldobs_12k\211209104213-14700.pth.optimizer
writer = SummaryWriter(comment="_" + "test_" + run_name)
# Use last action cache
USE_ACTION_CACHE = False

# Observation parameters (must match training parameters!)
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

####################################################

remote_client = FlatlandRemoteClient()

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)

if action_mask:
    set_action_size_reduced()
else:
    set_action_size_full()

def check_is_observation_valid(observation):
    return observation

def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
    return normalize_observation(observation, tree_depth, obs_type)


#New or old TreeObs
if obs_type == "NewTreeObs":
    tree_observation = NewTreeObs(max_depth=observation_tree_depth, predictor=predictor)
    print("\n>> NewTreeObs")
elif obs_type == "TreeObs":
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)
    print("\n>> TreeObs")
else:
    print("No observation was set")


# Calculate the state size given the depth of the tree observation and the number of features
n_features_per_node = tree_observation.observation_dim
n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = n_features_per_node * n_nodes

#####################################################################
# Main evaluation loop
#####################################################################
evaluation_number = 0
sum_normalized_reward = 0
while True:
    evaluation_number += 1

    # We use a dummy observation and call TreeObsForRailEnv ourselves when needed.
    # This way we decide if we want to calculate the observations or not instead
    # of having them calculated every time we perform an env step.
    time_start = time.time()
    observation, info = remote_client.env_create(obs_builder_object=DummyObservationBuilder())
    env_creation_time = time.time() - time_start

    if not observation:
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been
        # evaluated on all the required evaluation environments,
        # and hence it's safe to break out of the main evaluation loop.
        break

    print("Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)

    local_env = remote_client.env
    nb_agents = len(local_env.agents)
    max_nb_steps = local_env._max_episode_steps #Potential issues?
    if frame_skip:
        #local_env.reset(regenerate_schedule=True, regenerate_rail=True)
        local_env = RailEnvWrapper(local_env)
        SkipNoChoiceCellsWrapper(local_env, accumulate_skipped_rewards=False, discounting=0.0)

    tree_observation.set_env(local_env)
    tree_observation.reset()
 
    if load_policy == "PPO":
        policy = PPO(state_size, get_action_size(), use_replay_buffer=True)
        #policy = PPO(state_size, get_action_size(), use_replay_buffer=True, in_parameters=Namespace(**{'use_gpu': False})) ##True in  train, makes dif?
    else:
        policy = PPO(state_size, get_action_size())

    policy.load(checkpoint)

    policy.reset(local_env)
    observation = tree_observation.get_many(list(range(nb_agents)))

    print("Evaluation {}: {} agents in {}x{}".format(evaluation_number, nb_agents, local_env.width, local_env.height))

    # Now we enter into another infinite loop where we
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    steps = 0

    # Bookkeeping
    time_taken_by_controller = []
    time_taken_per_step = []

    # Action cache: keep track of last observation to avoid running the same inferrence multiple times.
    # This only makes sense for deterministic policies.
    agent_last_obs = {}
    agent_last_action = {}
    nb_hit = 0
    score = 0

    policy.start_episode(train=False)
    while True:
        try:
            #####################################################################
            # Evaluation of a single episode
            #####################################################################
            steps += 1
            obs_time, agent_time, step_time = 0.0, 0.0, 0.0
            no_ops_mode = False

            if not Deadlock.check_if_all_blocked(env=local_env): #not blocked
                time_start = time.time()
                action_dict = {}
                policy.start_step(train=False)
                for agent_handle in range(nb_agents):
                    if info['action_required'][agent_handle]:
                        if agent_handle in agent_last_obs and np.all(agent_last_obs[agent_handle] == observation[agent_handle]):
                            # cache hit
                            action = agent_last_action[agent_handle]
                            nb_hit += 1
                        else:
                            normalized_observation = get_normalized_observation(observation[agent_handle],
                                                                                observation_tree_depth,
                                                                                observation_radius=observation_radius)

                            action = policy.act(agent_handle, normalized_observation)
                        
                    else: 
                        action = 0



                    action_dict[agent_handle] = action

                    if USE_ACTION_CACHE:
                        agent_last_obs[agent_handle] = observation[agent_handle]
                        agent_last_action[agent_handle] = action
                    
                policy.end_step(train=False)
                agent_time = time.time() - time_start
                time_taken_by_controller.append(agent_time)

                time_start = time.time()
                try:
                    _, all_rewards, done, info = remote_client.env_step(map_actions(action_dict))
                    score += all_rewards[agent_handle]
                except:
                    print("failed to take env.step, non-deadlocked")
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

                time_start = time.time()
                observation = tree_observation.get_many(list(range(nb_agents)))
                obs_time = time.time() - time_start

            else:
                # Fully deadlocked: perform no-ops, optimization
                no_ops_mode = True

                time_start = time.time()
                try:
                    _, all_rewards, done, info = remote_client.env_step({})
                    score += all_rewards[agent_handle]
                except:
                    print("failed to take env.step, deadlocked")
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

            completions = 0
            for i_agent, agent in enumerate(local_env.agents):
                # manage the boolean flag to check if all agents are indeed done (or done_removed)
                if (agent.state in [TrainState.DONE]):
                    completions += 1
            
            if VERBOSE or done['__all__']:
                print(
                    "Step {}/{}\tAgents done: {}\t Obs time {:.3f}s\t Inference time {:.5f}s\t Step time {:.3f}s\t Cache hits {}\t No-ops? {}".format(
                        str(steps).zfill(4),
                        max_nb_steps,
                        completions,
                        obs_time,
                        agent_time,
                        step_time,
                        nb_hit,
                        no_ops_mode
                    ), end="\r")
                

            if done['__all__']:
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                print("Hit done all on step", steps)
                print("Reward : ", sum(list(all_rewards.values())))
                reward_test = sum(list(all_rewards.values()))
                break
        
        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive timeouts.
            print("Timeout! Will skip this episode and go to the next.", err)
            print("Hit Timeout on step", steps)
            break

    policy.end_episode(train=False)
    #Deadlock data
    deadlockData = Deadlock.deadlock_data(local_env)
    unfinished_agents_num = deadlockData[0]
    unfinished_not_deadlock = deadlockData[1]
    deadlocks = deadlockData[2]
    deadlock_percentage = deadlockData[2] / nb_agents
    completions_percentage = completions / nb_agents
  #  normalized_score = score / (max_nb_steps * nb_agents)
    normalized_score = reward_test / (max_nb_steps * nb_agents)
    
    
   # writer.add_scalar("testing/score", score, evaluation_number)
    writer.add_scalar("testing/normalized_reward", normalized_score, evaluation_number)
    writer.add_scalar("testing/num_agents", nb_agents, evaluation_number)
    writer.add_scalar("testing/deadlocks", deadlocks, evaluation_number)
    writer.add_scalar("testing/deadlock_percentage", deadlock_percentage, evaluation_number)
    writer.add_scalar("testing/done_agents", completions, evaluation_number)
    writer.add_scalar("testing/done_agents_percentage", completions_percentage, evaluation_number)
    writer.add_scalar("testing/score", reward_test, evaluation_number)
  #  writer.add_scalar("testing/normalized_reward_test", normalized_reward_test, evaluation_number)

    writer.add_scalar("testing/obs_time", obs_time, evaluation_number)
    writer.add_scalar("testing/agent_time", agent_time, evaluation_number)
    writer.add_scalar("testing/step_time", step_time, evaluation_number)
    writer.add_scalar("testing/done_agents_percentage", completions_percentage, evaluation_number)
    writer.flush()

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("=" * 100)
    sum_normalized_reward += normalized_score + 1

print("Sum normalized reward",sum_normalized_reward )

print("Evaluation of all environments complete!")
print(remote_client.submit())
