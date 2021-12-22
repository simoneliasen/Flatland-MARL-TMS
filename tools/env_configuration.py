
from flatland.envs.rail_env import RailEnv
from flatland.envs.malfunction_generators  import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator

def create_rail_env(env_params, tree_observation, test_seed = 0):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
   # max_rails_between_cities = env_params.max_rails_between_cities
   # max_rail_pairs_in_city = env_params.max_rail_pairs_in_city
    if test_seed == 0:
       seed = env_params.seed
    else:
        seed = test_seed

    malfunction_parameters = MalfunctionParameters(
                                            malfunction_rate = env_params.malfunction_rate,  # Rate of malfunction occurence
                                            min_duration = 20,  # Minimal duration of malfunction
                                            max_duration = 50  # Max duration of malfunction
                                            )
    #Train speeds
    train_speed_parameters = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            #max_rails_between_cities=max_rails_between_cities,
            #max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        line_generator=sparse_line_generator(train_speed_parameters),
        number_of_agents=n_agents,
        malfunction_generator = ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        remove_agents_at_target=True,
        random_seed=seed
    )