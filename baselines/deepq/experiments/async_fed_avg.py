import gym
import itertools
import sys  # for sys.argv
import random
import time  # for time.sleep
import argparse  # for parsing command line arguments... obviously
import configparser  # for parsing the config ini file
from datetime import datetime  # For generating timestamps for CSV files
import csv  # for writing to CSV files... obviously

import os  # for getting paths to this file
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))[0])[0])

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
python baselines\deepq\experiments\custom_cartpole.py
Synchronized?
https://stackoverflow.com/questions/42492589/distributed-tensorflow-on-distributed-data-synchronize-workers
https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
"""

# some parameters
seed = 1
env_name = "CartPole-v0"
max_reward = 200


def write_csv(file_name, episode, reward, avg_reward, t_global, factor, chief):
    if chief:
        try:
            with open(file_name + ".csv", 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                if episode == 1:
                    csv_writer.writerow(["episode", "reward", "avg_reward", "t_global", "factor"])
                csv_writer.writerow([episode, reward, avg_reward, t_global, factor])
        except PermissionError:
            print("Permission error. CSV write failed: [", episode, reward, avg_reward, t_global, factor, "]")


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
    config = configparser.ConfigParser()
    config.read(FLAGS.config_file)

    ps_hosts = FLAGS.ps_hosts if FLAGS.ps_hosts else config.get(FLAGS.config, 'ps_hosts').split(",")
    worker_hosts = FLAGS.worker_hosts if FLAGS.worker_hosts else config.get(FLAGS.config, 'worker_hosts').split(",")
    job = FLAGS.job_name
    task = FLAGS.task_index
    learning_rate = FLAGS.learning_rate if FLAGS.learning_rate else config.getfloat(FLAGS.config, 'learning_rate')
    batch_size = FLAGS.batch_size if FLAGS.batch_size else config.getint(FLAGS.config, 'batch_size')
    memory_size = FLAGS.memory_size if FLAGS.memory_size else config.getint(FLAGS.config, 'memory_size')
    target_update = FLAGS.target_update if FLAGS.target_update else config.getint(FLAGS.config, 'target_update')
    seed = FLAGS.seed if FLAGS.seed else config.getint(FLAGS.config, 'seed')
    comm_rounds = FLAGS.comm_rounds if FLAGS.comm_rounds else config.getint(FLAGS.config, 'comm_rounds')
    epochs = FLAGS.epochs if FLAGS.epochs else config.getint(FLAGS.config, 'epochs')
    backup = FLAGS.backup if FLAGS.backup else config.getint(FLAGS.config, 'backup')  # unused in async

    print("Config: (ps_hosts={}, worker_hosts={}, job_name={}, task_index={}, learning_rate={}, batch_size={}, "
          "memory_size={}, target_update={}, seed={}, comm_rounds={}, epochs={}, backup={})"
          .format(ps_hosts, worker_hosts, job, task, learning_rate, batch_size,
                  memory_size, target_update, seed, comm_rounds, epochs, backup))

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    chief = True if job == 'worker' and task == 0 else False
    print("/job:", job, "/task:", task, " - Chief: ", chief, sep='')

    # Create server
    server = tf.train.Server(cluster, job_name=job, task_index=task)

    # Set a unique random seed for each client
    seed += task
    random.seed(seed)
    run_code = datetime.now().strftime("%y%m%d-%H%M%S")
    run_code += "-" + env_name + "-p" + str(len(ps_hosts)) + "w" + str(len(worker_hosts)) + "-E" + str(epochs) + \
                "-b" + str(batch_size) + "-m" + str(memory_size) + "-N" + str(target_update)

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
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            chief=chief,
            server=server,
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(memory_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        if not chief:
            print("Worker {}/{} will sleep (3s) for chief to initialize variables".format(task+1, len(worker_hosts)))
            time.sleep(3)

        # Initialize the parameters and copy them to the target network.
        U.initialize(chief=chief)

        print("initialized variables, sleeping for 1 sec")
        time.sleep(1)

        t_global_old = update_weights()[0][0]
        update_target()
        t_start = 0
        factor = [[1]]

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

                # Append results to CSV file
                if len(episode_rewards) < 1:
                    avg_rew = 0
                else:
                    avg_rew = np.round(np.mean(np.array(episode_rewards[-100:])), 1)
                write_csv(run_code, len(episode_rewards), episode_rewards[-1], avg_rew, debug['t_global']()[0],
                          factor[0][0], chief)

                # Reset and prepare for next episode
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= max_reward
            if is_solved or t >= comm_rounds * epochs:  # TODO should be t_global, not t
                if chief:
                    print("Results saved in: ", run_code, ".csv", sep='')
                print("Converged after:  ", len(episode_rewards), "episodes")
                print("Agent total steps:", t)
                print("Global steps:     ", debug['t_global']()[0])
                return
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t >= 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    if t - t_start >= epochs:  # The number of local timesteps to calculate before averaging gradients
                        # print("t = {}, Updating global network (t_global = {})".format(t, debug['t_global']()[0]))

                        # Apply gradients to weights in PS
                        factor = global_opt([t - t_start], [t_global_old])

                        # Update the local weights with the new global weights from PS
                        t_global_old = update_weights()[0][0]

                        t_start = t

                # Update target network periodically.
                if t % target_update == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                last_rewards = episode_rewards[-101:-1]
                logger.record_tabular("steps", t)
                logger.record_tabular("global steps", debug['t_global']()[0])
                logger.record_tabular("communication rounds", debug['t_global']()[0] / epochs)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", np.round(np.mean(episode_rewards[-101:-1]), 4))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger.record_tabular("last gradient factor", np.round(factor, 4))
                logger.dump_tabular()
                print("[" + ''.join(
                    ['●' if x >= max_reward else str(int(np.floor(x / (max_reward/10)))) if x >= (max_reward/10) else '_' for x in last_rewards])
                      + "] (" + str(last_rewards.count(max_reward)) + "/", len(last_rewards), ")", sep='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.ini",
        help="Filename of config file. Overrides all arguments except 'job_name' and 'task_index'"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="async",
        help="Name of the section in the config file to read "
    )
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
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
        "--learning_rate",
        type=float,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Size of the sample used for the mini-batch fed to every learning step"
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        help="Max size of the experience replay memory"
    )
    parser.add_argument(
        "--target_update",
        type=int,
        help="Local timesteps between updates to the target Q-network"
    )
    # Flags for FedAvg algorithm hyperparameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for randomness (reproducibility)"
    )
    parser.add_argument(
        "--comm_rounds",
        type=int,
        help="Total number of communication rounds to execute before interrupting"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to run every communication round"
    )
    parser.add_argument(
        "--backup",
        type=int,
        help="Number of backup workers to use (implicitly sets n = total - b)"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("Unparsed:", unparsed)
    tf.app.run(main=main)
