import numpy as np

from observations.NewTreeObs import TreeObsForRailEnv

#Inspired by:
# https://github.com/mitchellgoffpc/flatland-training/blob/f81da955513267e104e651fa29687425adb23ad5/src/train.py



ACTIONS = ['L', 'F', 'R', 'B']

# Recursively create numpy arrays for each tree node
def create_tree_features(node, current_depth, max_depth, empty_node, data):
    if node == -np.inf:
        num_remaining_nodes = (4 ** (max_depth - current_depth + 1) - 1) // (4 - 1)
        data.extend([empty_node] * num_remaining_nodes)

    else:
        data.append(np.array(tuple(node)[:-2]))
        if node.childs:
            for direction in ACTIONS:
                create_tree_features(node.childs[direction], current_depth + 1, max_depth, empty_node, data)
    return data


# Normalize an observation to [0, 1] and then clip it to get rid of any infinite-valued features
def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    if fixed_radius > 0:
          max_obs = fixed_radius
    else: max_obs = np.max(obs[np.where(obs < 1000)], initial=1) + 1

    min_obs = np.min(obs[np.where(obs >= 0)], initial=max_obs) if normalize_to_range else 0

    if max_obs == min_obs:
          return np.clip(obs / max_obs, clip_min, clip_max)
    else: return np.clip((obs - min_obs) / np.abs(max_obs - min_obs), clip_min, clip_max)


# Normalize a tree observation
def normalize_observation(tree, max_depth, obs_type):
    if obs_type == "TreeObs":
        TreeObsForRailEnv.num_new_features = 0


   
    empty_node = np.array([0] * 6 + [np.inf] + [0] * (4+TreeObsForRailEnv.num_new_features)) # For policy networks #NY FEATURE: ÆNDRET HE 4 + 5 nye features = 9
    data = np.concatenate(create_tree_features(tree, 0, max_depth, empty_node, [])).reshape((-1, (11+TreeObsForRailEnv.num_new_features))) #NY FEATURE: ÆNDRET HE

    obs_data = norm_obs_clip(data[:,:6].flatten())
    distances = norm_obs_clip(data[:,6], normalize_to_range=True)
    agent_data = np.clip(data[:,7:].flatten(), -1, 1)

    return np.concatenate((obs_data, distances, agent_data))
