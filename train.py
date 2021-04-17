import argparse
import pickle
import random
import sys
from pathlib import Path

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import policy_saver
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common, nest_utils
from tqdm import tqdm

from envs import VectorEnv
import components
import utils

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        dummy_action_step = policy_step.PolicyStep(action=tf.constant([tf.int32.min]))
        dummy_time_step = ts.TimeStep(
            step_type=tf.constant([tf.int32.min]), reward=(np.nan * tf.ones(1)),
            discount=(np.nan * tf.ones(1)), observation=None)
        trajs = []
        for transition in random.sample(self.buffer, batch_size):
            traj1 = trajectory.from_transition(transition.time_step, transition.action_step, transition.next_time_step)
            traj2 = trajectory.from_transition(transition.next_time_step, dummy_action_step, dummy_time_step)
            trajs.append(nest_utils.unbatch_nested_tensors(nest_utils.stack_nested_tensors([traj1, traj2], axis=1)))
        return nest_utils.stack_nested_tensors(trajs)

    def __len__(self):
        return len(self.buffer)

def main(cfg):
    # Set up logging and checkpointing
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    # Create env
    env = utils.get_env_from_cfg(cfg)
    tf_env = components.get_tf_py_env(env, cfg.num_input_channels)

    # Agents
    epsilon = tf.Variable(1.0)
    agents = []
    for i, g in enumerate(cfg.robot_config):
        robot_type = next(iter(g))
        q_net = components.QNetwork(
            tf_env.observation_spec(), num_output_channels=VectorEnv.get_num_output_channels(robot_type))
        optimizer = keras.optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9)  # cfg.weight_decay is currently ignored
        agent_cls = dqn_agent.DdqnAgent if cfg.use_double_dqn else dqn_agent.DqnAgent
        agent = agent_cls(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=components.get_action_spec(robot_type),
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=epsilon,
            target_update_period=(cfg.target_update_freq // cfg.train_freq),
            td_errors_loss_fn=common.element_wise_huber_loss,
            gamma=cfg.discount_factors[i],
            gradient_clipping=cfg.grad_norm_clipping,
            train_step_counter=tf.Variable(0, dtype=tf.int64),  # Separate counter for each agent
        )
        agent.initialize()
        agent.train = common.function(agent.train)
        agents.append(agent)
    global_step = agents[0].train_step_counter

    # Replay buffers
    replay_buffers = [ReplayBuffer(cfg.replay_buffer_size) for _ in agents]

    # Checkpointing
    timestep_var = tf.Variable(0, dtype=tf.int64)
    agent_checkpointer = common.Checkpointer(
        ckpt_dir=str(checkpoint_dir / 'agents'), max_to_keep=5, agents=agents, timestep_var=timestep_var)
    agent_checkpointer.initialize_or_restore()
    if timestep_var.numpy() > 0:
        checkpoint_path = checkpoint_dir / 'checkpoint_{:08d}.pkl'.format(timestep_var.numpy())
        with open(checkpoint_path, 'rb') as f:
            replay_buffers = pickle.load(f)

    # Logging
    train_summary_writer = tf.summary.create_file_writer(str(log_dir / 'train'))
    train_summary_writer.set_as_default()

    time_step = tf_env.reset()
    learning_starts = round(cfg.learning_starts_frac * cfg.total_timesteps)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    start_timestep = timestep_var.numpy()
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Set exploration epsilon
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        epsilon.assign(exploration_eps)

        # Run one collect step
        transitions_per_buffer = tf_env.pyenv.envs[0].store_time_step(time_step)
        robot_group_index = tf_env.pyenv.envs[0].current_robot_group_index()
        action_step = agents[robot_group_index].collect_policy.action(time_step)
        time_step = tf_env.step(action_step.action)

        # Store experience in buffers
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                replay_buffers[i].push(*transition)

        # Train policies
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            for i, agent in enumerate(agents):
                experience = replay_buffers[i].sample(cfg.batch_size)
                agent.train(experience)

        # Logging
        if tf_env.pyenv.envs[0].done():
            info = tf_env.pyenv.envs[0].get_info()
            tf.summary.scalar('timesteps', timestep + 1, global_step)
            tf.summary.scalar('steps', info['steps'], global_step)
            tf.summary.scalar('total_cubes', info['total_cubes'], global_step)

        # Checkpointing
        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            # Save agents
            timestep_var.assign(timestep + 1)
            agent_checkpointer.save(timestep + 1)

            # Save replay buffers
            checkpoint_path = checkpoint_dir / 'checkpoint_{:08d}.pkl'.format(timestep + 1)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(replay_buffers, f)
            cfg.checkpoint_path = str(checkpoint_path)
            utils.save_config(log_dir / 'config.yml', cfg)

            # Remove old checkpoints
            checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.pkl'))
            checkpoint_paths.remove(checkpoint_path)
            for old_checkpoint_path in checkpoint_paths:
                old_checkpoint_path.unlink()

    # Export trained policies
    policy_dir = checkpoint_dir / 'policies'
    for i, agent in enumerate(agents):
        policy_saver.PolicySaver(agent.policy).save(str(policy_dir / 'robot_group_{:02}'.format(i + 1)))
    cfg.policy_path = str(policy_dir)
    utils.save_config(log_dir / 'config.yml', cfg)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    config_path = parser.parse_args().config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is not None:
        config_path = utils.setup_run(config_path)
        main(utils.load_config(config_path))
