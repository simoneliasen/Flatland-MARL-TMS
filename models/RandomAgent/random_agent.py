import numpy as np
#import time
import argparse
from flatland.envs.rail_env import RailEnv #, RailEnvactions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators  import malfunction_from_params, MalfunctionParameters,malfunction_from_file,ParamMalfunctionGen
from flatland.evaluators.client import FlatlandRemoteClient

#####################################################################
# Passing Arguments
#####################################################################

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--render', default=False, action='store_true') #python starter.py --render <---- to Render enviroment
parser.add_argument('--submit', default=True, action='store_true') #python starter.py --submit <---- to Submit online
arguments = parser.parse_args()

# Turns on render-option for local viewing and sets submission off.
if arguments.render:
    arguments.submit = False

#Establish connection to remote client for submission (default)
if arguments.submit:
    remote_client = FlatlandRemoteClient() 


#####################################################################
# Malfunctions & Speed Parameters
#####################################################################


stochastic_data = MalfunctionParameters(malfunction_rate = 1/200,  # Rate of malfunction occurence
                                        min_duration = 20,  # Minimal duration of malfunction
                                        max_duration = 50  # Max duration of malfunction
                                        )

speed_ration_map = {1.: 1,  # Fast passenger train
                    1. / 2.: 0,  # Fast freight train
                    1. / 3.: 0,  # Slow commuter train
                    1. / 4.: 0}  # Slow freight train

# Generates new rail networks on each reset
rail_generator = sparse_rail_generator(#max_num_cities=5,
                                       #seed=10,
                                       grid_mode=False,
                                       #max_rails_between_cities=1,
                                      # max_rail_pairs_in_city=1.5
                                       )

#Generates new rail networks on each reset,
line_generator = sparse_line_generator(speed_ration_map)


#####################################################################
# Initialize Flatland Environment
#####################################################################


# Parameters for the Environment
x_dim = np.random.randint(30, 35)
y_dim = np.random.randint(30, 35)
n_agents = np.random.randint(3, 8)
observation_tree_depth = 3
obs_builder_object = TreeObsForRailEnv(max_depth=observation_tree_depth)


local_env = RailEnv(width=x_dim,
                    height=y_dim,
                    rail_generator=rail_generator,
                    line_generator=line_generator, 
                    #obs_builder_object=GlobalObsForRailEnv(), #Global observation
                    obs_builder_object= obs_builder_object, #Tree observation
                    malfunction_generator=ParamMalfunctionGen(stochastic_data),
                    remove_agents_at_target=True,
                    number_of_agents=n_agents
                    #random_seed=100 #if we want to test the same map
                    )

# Calculates state and action sizes
n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = obs_builder_object.observation_dim * n_nodes
action_size = 5 # Action-space/Action_size (5) = same as: local_env.action_space[0]


#####################################################################
# Tool For Rendering the enviroment
#####################################################################


env_renderer = RenderTool(
			local_env,
            gl="PGL",
			agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
			show_debug=True,
			screen_height=1080,
			screen_width=1920
            )


#####################################################################
# The Controller of agents
#####################################################################


class RandomController:
    #Initialized by giving an action_space(N,S,W,E,HALT)
    def __init__(self, action_size):
        self.action_size = action_size

    #Takes random action from action_space
    def act(self, obs):
        actions = dict()
        for agent_handle, observation in enumerate(obs):
            action = np.random.randint(self.action_size)
            actions.update({agent_handle: action})
        return actions


#####################################################################
# Main loop
#####################################################################


episode_number = 0

while True:

    print("==============")
    episode_number += 1
    print("[INFO] EPISODE_number_START : {}".format(episode_number))
    
    if arguments.render:
        obs, info = local_env.reset()
        controller = RandomController(local_env.action_space[0])
        env_renderer.reset()

    if arguments.submit:
        obs, info = remote_client.env_create(obs_builder_object=obs_builder_object)
        controller = RandomController(remote_client.env.action_space[0]) 
        
    if not obs:
        """
        The remote env returns False as the first obs
        when it is done evaluating all the individual episode_numbers 
        """
        print("[INFO] DONE ALL, BREAKING")
        break


    while True:

        if arguments.render: #Render if selected
            env_renderer.render_env(show=True, show_observations=True, show_predictions=True)

        action = controller.act(obs) 

        try:
            if arguments.render:
                next_obs, all_rewards, done, info = local_env.step(action) #execute random action
            if arguments.submit:
                next_obs, all_rewards, done, info = remote_client.env_step(action) 
            #time.sleep(0.3) #For testing

        except:
            print("[ERR] DONE BUT step() CALLED")
       
        if (True):  # debug
            print("-----")
            # print(done)
            print("[DEBUG] REW: ", all_rewards)
         
        # break
        if done['__all__']:
            print("[INFO] EPISODE_number_DONE : ", episode_number)
            print("[INFO] FINAL REW: ", all_rewards)
            print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
            break


print("Evaluation Complete...")

if arguments.submit:
    print(remote_client.submit()) 


