import argparse
import components
import utils

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    print(config_path)
    cfg = utils.load_config(config_path)

    # Create env
    env = utils.get_env_from_cfg(cfg, show_gui=True)
    tf_env = components.get_tf_py_env(env, cfg.num_input_channels)

    # Load policies
    policies = components.load_policies(cfg)

    # Run policies
    time_step = tf_env.reset()
    while True:
        robot_group_index = tf_env.pyenv.envs[0].current_robot_group_index()
        action_step = policies[robot_group_index].action(time_step)
        time_step = tf_env.step(action_step.action)

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
main(parser.parse_args())
