import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

# some parameters
global_update = 50000


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        chief = True
        # Create all the functions necessary to train the model
        act, train, global_opt, update_target, update_weights, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            chief=chief,
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        t_global_old = update_weights()[0][0]
        update_target()
        # grad_sum = 0  # TODO The fact that this turns into a np.matrix upon add-assign (+=) is why python is good :)
        t_start = 0

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                if len(episode_rewards) % 10 == 0:
                    env.render()
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t >= 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    # td_error, gradient = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # grad_sum += np.matrix([g for g, _ in gradient])

                    # TODO What should trigger this global opt? just every n iterations? every ep?
                    if t - t_start >= 100:  # The number of local timesteps to calculate before averaging gradients
                        # print("t = {}, Updating global network (t_global = {})".format(t, debug['t_global']()[0]))

                        # Turn gradients back to list
                        # grad_sum = grad_sum.tolist()[0]

                        # Apply gradients to weights in PS
                        # global_opt(grad_sum)
                        global_opt([t - t_start], [t_global_old])

                        # Update the local weights with the new global weights from PS
                        t_global_old = update_weights()[0][0]

                        # TODO Update the target here too? Or just continue its normal cycle but with very old weights
                        # Reset all variables related to the global weights cycle and update
                        # grad_sum = 0
                        t_start = t

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
