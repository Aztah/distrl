import gym
import itertools
import sys  # for sys.argv
import random
import time  # for time.sleep
import argparse  # for parsing command line arguments... obviously
from datetime import datetime  # For generating timestamps for CSV files
import csv  # for writing to CSV files... obviously

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

"""
activate py36
cd PycharmProjects\distrl
python baselines\deepq\experiments\custom_cartpole.py 0
926 926
"""

# some parameters
seed = 1
env_name = "CartPole-v0"


def write_csv(file_name, episode, reward, avg_reward, t_global, chief):
    if chief:
        try:
            with open(file_name + ".csv", 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                if episode == 1:
                    csv_writer.writerow(["episode", "reward", "avg_reward", "t_global"])
                csv_writer.writerow([episode, reward, avg_reward, t_global])
        except PermissionError:
            print("Permission error. Could not write to CSV: [", episode, reward, avg_reward, t_global, "]")

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        return out


def main(_):

    print("Used flags:", FLAGS)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    job = FLAGS.job_name
    task = FLAGS.task_index

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    chief = True if job == 'worker' and task == 0 else False
    print("/job:", job, "/task:", task, " - Chief: ", chief, sep='')

    # Create server
    server = tf.train.Server(cluster, job_name=job, task_index=task)

    # Set a unique random seed for each client
    seed = task
    random.seed(seed)
    run_code = datetime.now().strftime("%y%m%d-%H%M%S")
    run_code += "-" + env_name + "-p" + str(len(ps_hosts)) + "w" + str(len(worker_hosts)) + "-E" + str(FLAGS.epochs) + \
                "-b" + str(FLAGS.batch_size) + "-m" + str(FLAGS.memory_size) + "-N" + str(FLAGS.target_update)

    print("Run code:", run_code)

    # Start parameter servers
    if job == 'ps':
        server.join()

    # Start training
    with U.make_session(num_cpu=4, target=server.target):
        # Create the environment
        env = gym.make(env_name)
        env.seed(seed)
        tf.set_random_seed(seed)

        # Create all the functions necessary to train the model
        act, train, global_opt, update_target, update_weights, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            chief=chief,
            server=server,
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(FLAGS.memory_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        if not chief:
            print("Worker {}/{} will sleep (3s) for chief to initialize variables".format(task+1, len(worker_hosts)))
            time.sleep(3)

        # Initialize the parameters and copy them to the target network.
        U.initialize(chief=chief)

        print("initialized variables, sleeping for 3 sec")
        time.sleep(3)

        t_global_old = update_weights()[0][0]
        update_target()
        # grad_sum = 0  # TODO The fact that this turns into a np.matrix upon add-assign (+=) is why python is good :)
        t_start = 0
        factor = 1

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
                # if len(episode_rewards) % 10 == 0:
                #     env.render()

                # Reset and prepare for next episode
                obs = env.reset()
                episode_rewards.append(0)

                # Append results to CSV file
                if len(episode_rewards) < 1:
                    avg_rew = 0
                else:
                    avg_rew = np.round(np.mean(np.array(episode_rewards[-101:-1])), 1)
                write_csv(run_code, len(episode_rewards), episode_rewards[-2], avg_rew, debug['t_global']()[0], chief)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved or t >= FLAGS.comm_rounds * FLAGS.epochs:
                if chief:
                    print("Results saved in: ", run_code, ".csv", sep='')
                print("Converged after:  ", len(episode_rewards), "episodes")
                print("Agent total steps:", t)
                print("Global steps:     ", debug['t_global']()[0])
                return
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t >= 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(FLAGS.batch_size)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    # td_error, gradient = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # grad_sum += np.matrix([g for g, _ in gradient])

                    if t - t_start >= FLAGS.epochs:  # The number of local timesteps to calculate before averaging gradients
                        # print("t = {}, Updating global network (t_global = {})".format(t, debug['t_global']()[0]))

                        # Turn gradients back to list
                        # grad_sum = grad_sum.tolist()[0]

                        # Apply gradients to weights in PS
                        # global_opt(grad_sum)
                        factor = global_opt([t - t_start], [t_global_old])

                        # Update the local weights with the new global weights from PS
                        t_global_old = update_weights()[0][0]

                        # grad_sum = 0
                        t_start = t

                # Update target network periodically.
                if t % FLAGS.target_update == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                last_rewards = episode_rewards[-101:-1]
                logger.record_tabular("steps", t)
                logger.record_tabular("global steps", debug['t_global']()[0])
                logger.record_tabular("communication rounds", debug['t_global']()[0] / FLAGS.epochs)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", np.round(np.mean(episode_rewards[-101:-1]), 4))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger.record_tabular("last gradient factor", np.round(factor, 4))
                logger.dump_tabular()
                print("[" + ''.join(
                    ['●' if x >= 200 else str(int(np.floor(x / 20))) if x >= 20 else '_' for x in last_rewards])
                      + "] (" + str(last_rewards.count(200)) + "/", len(last_rewards), ")", sep='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="localhost:13337",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="localhost:13338,localhost:13339,localhost:13340",
      help="Comma-separated list of hostname:port pairs"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--job_name",
      type=str,
      default="worker",
      help="One of 'ps', 'worker'"
    )
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    # Flags for the Q-learning hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Size of the sample used for the mini-batch fed to every learning step"
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=50000,
        help="Max size of the experience replay memory"
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=1000,
        help="Local timesteps between updates to the target Q-network"
    )
    # Flags for FedAvg algorithm hyperparameters
    parser.add_argument(
        "--comm_rounds",
        type=int,
        default=500000,
        help="Total number of communication rounds to execute before interrupting"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run every communication round"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)
