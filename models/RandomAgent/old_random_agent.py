from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
import numpy as np
np.random.seed(0)

#Remote connection to submission enviroment
remote_client = FlatlandRemoteClient()


#Agent-Controller that takes random actions, out the distinct(5) actionspace
def my_controller(obs, _env):
    _action = {}
    for _idx, _ in enumerate(_env.agents):
        _action[_idx] = np.random.randint(0, 5)
    return _action


# my_observation_builder = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
# max_depth=3, predictor=ShortestPathPredictorForRailEnv())
my_observation_builder = GlobalObsForRailEnv()



episode = 0

while True:

    print("==============")
    episode += 1
    print("[INFO] EPISODE_START : {}".format(episode))
    # NO WAY TO CHECK service/self.evaluation_done in client

    obs, info = remote_client.env_create(obs_builder_object=my_observation_builder)
    if not obs:
        """
        The remote env returns False as the first obs
        when it is done evaluating all the individual episodes 
        """
        print("[INFO] DONE ALL, BREAKING")
        break

    while True:
        action = my_controller(obs, remote_client.env)
        try:
            observation, all_rewards, done, info = remote_client.env_step(action)
        except:
            print("[ERR] DONE BUT step() CALLED")

        if (True):  # debug
            print("-----")
            # print(done)
            print("[DEBUG] REW: ", all_rewards)
        # break
        if done['__all__']:
            print("[INFO] EPISODE_DONE : ", episode)
            print("[INFO] FINAL REW: ", all_rewards)
            print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
            break

print("Evaluation Complete...")
print(remote_client.submit())
