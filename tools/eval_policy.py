from tools.agent_action_config import map_actions
from tools.deadlock_check import Deadlock 
import numpy as np
from tools.env_configuration import create_rail_env
from tools.normalize_observation import normalize_observation
from flatland.envs.agent_utils import EnvAgent, TrainState

def eval_policy(env, policy, tree_observation, train_params, obs_params, env_params, tree_obs):
    n_eval_episodes = train_params.n_evaluation_episodes
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):

        env.n_agents = env_params[episode_idx]["n_agents"]
        env.x_dim = env_params[episode_idx]["x_dim"]
        env.y_dim = env_params[episode_idx]["y_dim"]
        env.n_cities = env_params[episode_idx]["n_cities"]
        env.malfunction_rate = env_params[episode_idx]["malfunction_rate"]
        env.seed = env_params[episode_idx]["seed"]
      
        env = create_rail_env(env, tree_observation)
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        max_steps = env._max_episode_steps
        action_dict = dict()
        agent_obs = [None] * env.get_num_agents()    
        score = 0.0
        final_step = 0

        print("\nEvaluating ", env.get_num_agents(), env.width, env.height)

        for step in range(max_steps):
            deadlockData = Deadlock.deadlock_data(env)
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], max_depth=2, obs_type=tree_obs)
                    
                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent, agent_obs[agent])
                    
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        #Deadlock data on end episode
        deadlocked_agents_list = deadlockData[4]
        single_agent_deadlock = agent in deadlocked_agents_list
        unfinished_agents_num = deadlockData[0]
        unfinished_not_deadlock = deadlockData[1]
        deadlocks = deadlockData[2]
        is_deadlock = deadlockData[3]
        deadlock_data = unfinished_agents_num, unfinished_not_deadlock, deadlocks, is_deadlock  

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum([agent.state == TrainState.DONE for agent in env.agents])
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return deadlock_data, scores, completions, nb_steps
 
