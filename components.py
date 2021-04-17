from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents import specs
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

from envs import VectorEnv
from networks import fcn

def get_action_spec(robot_type):
    return tensor_spec.from_spec(specs.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=(VectorEnv.get_action_space(robot_type) - 1), name='action'))

def get_tf_py_env(env, num_input_channels):
    return tf_py_environment.TFPyEnvironment(VectorPyEnv(env, num_input_channels))

def load_policies(cfg):
    policies = []
    for i in range(len(cfg.robot_config)):
        policy_dir = Path(cfg.checkpoint_dir) / 'policies' / 'robot_group_{:02}'.format(i + 1)
        policy = tf.saved_model.load(str(policy_dir))
        policies.append(policy)
    return policies

class VectorPyEnv(py_environment.PyEnvironment):
    def __init__(self, env, num_input_channels):
        super().__init__()
        self._env = env
        self._done = True
        self._info = None
        self._observation_spec = specs.ArraySpec(shape=(self._env.get_state_width(), self._env.get_state_width(), num_input_channels), dtype=np.float32, name='observation')
        self._action_spec = specs.ArraySpec(shape=(), dtype=np.int32)  # Not used
        self._empty_observation = np.zeros(self._observation_spec.shape, dtype=self._observation_spec.dtype)
        self._ready_state = None
        self._prev_time_steps = None
        self._prev_actions = None

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _step(self, action):
        if self._done:
            return self.reset()

        a = action
        action = [[None for _ in g] for g in self._ready_state]
        for i, g in enumerate(self._ready_state):
            for j, s in enumerate(g):
                if s is not None:
                    action[i][j] = a
                    self._prev_actions[i][j] = policy_step.PolicyStep(action=tf.constant([a]))
        self._ready_state, reward, self._done, self._info = self._env.step(action)

        if self._done:
            # Arbitrarily return the first robot's reward
            return ts.termination(self._empty_observation, reward[0][0])

        for i, g in enumerate(self._ready_state):
            for j, s in enumerate(g):
                if s is not None:
                    return ts.transition(s, reward[i][j])

    def _reset(self):
        self._done = False
        self._ready_state = self._env.reset()
        self._prev_time_steps = [[None for _ in g] for g in self._ready_state]
        self._prev_actions = [[None for _ in g] for g in self._ready_state]
        return ts.restart(self._ready_state[0][0])

    def current_robot_group_index(self):
        for i, g in enumerate(self._ready_state):
            for s in g:
                if s is not None:
                    return i
        return 0  # Episode is over

    def done(self):
        return self._done

    def get_info(self):
        return self._info

    def store_time_step(self, time_step):
        transitions_per_buffer = [[] for g in self._ready_state]
        for i, g in enumerate(self._ready_state):
            for j, s in enumerate(g):
                if s is not None:
                    if self._prev_time_steps[i][j] is not None:
                        prev_time_step = self._prev_time_steps[i][j]
                        action = self._prev_actions[i][j]
                        transitions_per_buffer[i].append((prev_time_step, action, time_step))
                    self._prev_time_steps[i][j] = time_step
        return transitions_per_buffer

class QNetwork(network.Network):
    def __init__(self, input_tensor_spec, num_output_channels, name=None):
        super().__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        inputs = keras.Input(shape=input_tensor_spec.shape)
        x = fcn(inputs, num_output_channels=num_output_channels)
        x = keras.layers.Permute((3, 1, 2))(x)
        outputs = keras.layers.Flatten()(x)
        self.model = keras.Model(inputs, outputs)

    def call(self, inputs, step_type=None, network_state=(), training=False):
        q_value = self.model(inputs, training=training)
        return q_value, network_state
