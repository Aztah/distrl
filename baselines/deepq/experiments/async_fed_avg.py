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


def write_csv(file_name, episode, reward, avg_reward, t_global, dt, chief):
    if chief:
        try:
            with open(file_name + ".csv", 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                if episode == 1:
                    csv_writer.writerow(["episode", "reward", "avg_reward", "t_global", "dt"])
                csv_writer.writerow([episode, reward, avg_reward, t_global, dt])
        except PermissionError:
            print("Permission error. CSV write failed: [", episode, reward, avg_reward, t_global, dt, "]")


def write_csv_final(file_name, final_episode, round_log):
    new_filename = file_name + "=" + str(final_episode) + "ep.csv"
    with open(file_name + ".csv", 'r') as infile, open(new_filename, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=';')
        i = 0
        for row in reader:
            writer.writerow(row[0].split(";") + ([] if i >= len(round_log) else round_log[i]))
            i += 1

        for a in round_log[i:]:
            writer.writerow([None, None, None, None, None] + a)
    os.remove(file_name + ".csv")
    print("Results saved in:  ", new_filename, sep='')


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

    ps_hosts = config.get(FLAGS.config, 'ps_hosts').split(",")
    worker_hosts = config.get(FLAGS.config, 'worker_hosts').split(",")
    job = FLAGS.job_name
    task = FLAGS.task_index
    learning_rate = config.getfloat(FLAGS.config, 'learning_rate')
    batch_size = config.getint(FLAGS.config, 'batch_size')
    memory_size = config.getint(FLAGS.config, 'memory_size')
    target_update = config.getint(FLAGS.config, 'target_update')
    seed = FLAGS.seed if FLAGS.seed else config.getint(FLAGS.config, 'seed')
    max_comm_rounds = config.getint(FLAGS.config, 'comm_rounds')
    epochs = config.getint(FLAGS.config, 'start_epoch')
    end_epoch = config.getint(FLAGS.config, 'end_epoch')
    epoch_decay = config.getint(FLAGS.config, 'epoch_decay')
    # epoch_decay_rate = (epochs - end_epoch) / epoch_decay
    epoch = LinearSchedule(epoch_decay, end_epoch, epochs)
    end_alpha = config.getfloat(FLAGS.config, 'end_alpha')
    alpha_decay = config.getint(FLAGS.config, 'alpha_decay')
    alpha = LinearSchedule(alpha_decay, end_alpha)
    backup = config.getint(FLAGS.config, 'backup')  # unused in async

    print("Config:\nps_hosts={}\nworker_hosts={}\njob_name={}\ntask_index={}\nlearning_rate={}\n"
          "batch_size={}\nmemory_size={}\ntarget_update={}\nseed={}\ncomm_rounds={}\nepochs={}\n"
          "end_epoch={}\nepoch_decay={}\nend_alpha={}\nalpha_decay={}backup={}"
          .format(ps_hosts, worker_hosts, job, task, learning_rate, batch_size, memory_size, target_update,
                  seed, max_comm_rounds, epochs, end_epoch, epoch_decay, end_alpha, alpha_decay, backup))

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    chief = True if job == 'worker' and task == 0 else False
    print("/job:", job, "/task:", task, " - Chief: ", chief, sep='')

    # Create server
    server = tf.train.Server(cluster, job_name=job, task_index=task)

    # Set a unique random seed for each client
    seed += task
    random.seed(seed)

    run_code = "{}-{}-p{}w{}-E{}-b{}-m{}-N{}-lr{}-B{}".\
        format(datetime.now().strftime("%y%m%d-%H%M%S"), env_name, len(ps_hosts), len(worker_hosts),
               epochs, batch_size, memory_size, target_update, learning_rate, backup)

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
            # optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
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
        exp_gen = 1000  # For how many timesteps sould we only generate experience (not train)
        t_start = exp_gen
        comm_rounds = 0
        dt = [[0]]
        round_log = [[None, "comm_rounds", "t", "staleness", "epoch"]]

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
                avg_rew = np.round(np.mean(np.array(episode_rewards[-100:])), 1)
                write_csv(run_code, len(episode_rewards), episode_rewards[-1], avg_rew, debug['t_global']()[0],
                          dt[0][0], chief)

                # Reset and prepare for next episode
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= max_reward
            if is_solved or comm_rounds >= max_comm_rounds:
                if chief:
                    write_csv_final(run_code, str(len(episode_rewards)), round_log)
                print("Converged after:  ", len(episode_rewards), "episodes")
                print("Agent total steps:", t)
                print("Global steps:     ", debug['t_global']()[0])
                return
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t >= exp_gen:
                # if t >= batch_size:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    if t - t_start >= np.round(epoch.value(comm_rounds)):  # The number of local timesteps to calculate before averaging gradients
                        # print("t = {}, Updating global network (t_global = {})".format(t, debug['t_global']()[0]))

                        # Apply gradients to weights in PS
                        dt = global_opt([t - t_start], [t_global_old], [alpha.value(comm_rounds)])

                        # Update the local weights with the new global weights from PS
                        t_global_old = update_weights()[0][0]

                        comm_rounds += 1
                        round_log.append([None])
                        round_log[-1].append(comm_rounds)
                        round_log[-1].append(t)
                        round_log[-1].append(dt[0][0])
                        round_log[-1].append(epoch.value(comm_rounds))

                        t_start = t
                        # epochs = end_epoch if epochs <= end_epoch else epochs - epoch_decay_rate

                # Update target network periodically.
                if t % target_update == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                last_rewards = episode_rewards[-101:-1]
                logger.record_tabular("steps", t)
                logger.record_tabular("global steps", debug['t_global']()[0])
                logger.record_tabular("communication rounds", comm_rounds)
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
        help="Filename of config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DEFAULT",
        help="Name of the section in the config file to read "
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
    # Flags for FedAvg algorithm hyperparameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for randomness (reproducibility)"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("Unparsed:", unparsed)
    tf.app.run(main=main)
