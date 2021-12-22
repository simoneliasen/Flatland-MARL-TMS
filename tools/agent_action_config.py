from flatland.envs.rail_env import RailEnvActions

# global action size
global _agent_action_config_action_size
_agent_action_config_action_size = 5


def get_flatland_full_action_size():
    # The action space of flatland is 5 discrete actions
    return 5


def set_action_size_full():
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    _agent_action_config_action_size = 5


def set_action_size_reduced():
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    _agent_action_config_action_size = 4


def get_action_size():
    global _agent_action_config_action_size
    # The agents (DDDQN, PPO, ... ) have this actions space
    return _agent_action_config_action_size


def map_actions(actions):
    # Map the
    if get_action_size() != get_flatland_full_action_size():
        for key in actions:
            value = actions.get(key, 0)
            actions.update({key: map_action(value)})
    return actions


def map_action_policy(action):
    if get_action_size() != get_flatland_full_action_size():
        return action - 1
    return action


def map_action(action):
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == 0:
        return RailEnvActions.MOVE_LEFT
    if action == 1:
        return RailEnvActions.MOVE_FORWARD
    if action == 2:
        return RailEnvActions.MOVE_RIGHT
    if action == 3:
        return RailEnvActions.STOP_MOVING


def map_rail_env_action(action):
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == RailEnvActions.MOVE_LEFT:
        return 0
    elif action == RailEnvActions.MOVE_FORWARD:
        return 1
    elif action == RailEnvActions.MOVE_RIGHT:
        return 2
    elif action == RailEnvActions.STOP_MOVING:
        return 3
    # action == RailEnvActions.DO_NOTHING:
    return 3
