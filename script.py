from flatland.evaluators.service import FlatlandRemoteEvaluationService
from flatland.evaluators import messages


if __name__ == "__main__":

    grader = FlatlandRemoteEvaluationService(
        test_env_folder="/mnt/c/Users/simon/Desktop/flatland-starter-kit/debug-environments",
        flatland_rl_service_id='FLATLAND_RL_SERVICE_ID',
        visualize=False,
        video_generation_envs=["Test_0/Level_0.pkl"],
        report=None,
        verbose=False,
        action_dir=None,
        episode_dir=None,
        merge_dir=None,
        use_pickle=False,
        shuffle=False,
        missing_only=False,
        result_output_path=None,
        disable_timeouts=False
    )
    result = grader.run()

    if result['type'] == messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
    elif result['type'] == messages.FLATLAND_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        # Evaluation failed
        print("Evaluation Failed : ", result['payload'])