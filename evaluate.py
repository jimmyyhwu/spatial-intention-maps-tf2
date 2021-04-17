import argparse

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import tensorflow as tf
import components
import utils

# Use single thread for tensorflow cpu work
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def run_eval(cfg, num_episodes=20):
    random_seed = 0

    # Create env
    env = utils.get_env_from_cfg(cfg, random_seed=random_seed, use_egl_renderer=False)
    tf_env = components.get_tf_py_env(env, cfg.num_input_channels)

    # Load policies
    policies = components.load_policies(cfg)

    # Run policies
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    time_step = tf_env.reset()
    while True:
        robot_group_index = tf_env.pyenv.envs[0].current_robot_group_index()
        action_step = policies[robot_group_index].action(time_step)
        time_step = tf_env.step(action_step.action)

        info = tf_env.pyenv.envs[0].get_info()
        data[episode_count].append({
            'simulation_steps': info['simulation_steps'],
            'cubes': info['total_cubes'],
        })
        if tf_env.pyenv.envs[0].done():
            episode_count += 1
            print('Completed {}/{} episodes'.format(episode_count, num_episodes))
            if episode_count >= num_episodes:
                break
            time_step = tf_env.reset()

    env.close()

    return data

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    cfg = utils.load_config(config_path)
    eval_dir = utils.get_eval_dir()
    eval_path = eval_dir / '{}.npy'.format(cfg.run_name)
    data = run_eval(cfg)
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_path, np.array(data, dtype=object))
    print(eval_path)

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
main(parser.parse_args())
