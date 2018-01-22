import gym
import itertools
import sys  # for sys.argv
import random
import time  # for time.sleep
import argparse  # for parsing command line arguments... obviously
import configparser  # for parsing the config ini file
from datetime import datetime  # For generating timestamps for CSV files
import csv  # for writing to CSV files... obviously
import bisect  # for inserting into sorted lists
import statistics  # for averaging and stuff

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


def write_csv(run_code, log, comm_rounds=False):
    file_name = run_code + ("cr" if comm_rounds else "") + ".csv"
    if log is not None:
        try:
            with open(file_name, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                csv_writer.writerow(log)
        except PermissionError:
            print("Permission error. CSV write failed:", log)


def write_csv_final(file_name, final_episode, worker_hosts=None, chief=False, comm_rounds=0, mute=False):
    new_filename = file_name + "=" + str(final_episode) + "ep.csv"
    os.rename(file_name + ".csv", new_filename)
    # with open(file_name + ".csv", 'r', newline='') as infile, open(new_filename, 'w', newline='') as outfile:
    #     reader = csv.reader(infile, delimiter=';')
    #     writer = csv.writer(outfile, delimiter=';')
    #     i = 0
    #     for row in reader:
    #         writer.writerow(row + ([] if i >= len(round_log) else round_log[i]))
    #         i += 1
    #
    #     for a in round_log[i:]:
    #         writer.writerow([None, None, None, None] + a)
    # os.remove(file_name + ".csv")
    if chief:
        if all([host.find("localhost") >= 0 for host in worker_hosts]):
            data = []
            f1 = file_name.split("(w")[0]
            files = []
            if not mute:
                print("All localhost. Chief combining files")
            while len(files) < len(worker_hosts):
                files = list(filter(lambda f: f.find(f1) >= 0 and f.find(")=") >= 0, os.listdir('.')))
            if not mute:
                print("Files to combine:", files)
            for file in files:
                buffer = []
                # TODO fix permission error: PermissionError: [Errno 13] Permission denied
                for attempt in range(100):
                    try:
                        with open(file, 'r', newline='') as infile:
                            reader = csv.reader(infile, delimiter=';')
                            buffer = [row for row in reader]
                        bisect.insort_left(data, buffer)
                    except PermissionError as e:
                        print("Could not open file ", file, " Permission error(", attempt, "):", e.strerror, sep='')
                        time.sleep(5)
                        continue
                    else:
                        break
                else:
                    print("All failed. Some files will not be combined.")
            data_len = [len(x) for x in data]
            if not mute:
                print("Data of length", data_len, "\n", data)
            summary_name = "{}-avg-{}-med-{}-sdv-{}-min-{}-max-{}-cr-{}.csv"\
                .format(file_name.split("(")[0], round(statistics.mean(data_len)),
                        round(statistics.median(data_len)),
                        (round(statistics.stdev(data_len), 1) if len(data_len) > 1 else 0),
                        min(data_len), max(data_len), int(comm_rounds))
            with open(summary_name, 'w', newline='') as csv_file:
                i = 0
                csv_writer = csv.writer(csv_file, delimiter=';')
                while i < max(data_len):
                    writing = []
                    for j in range(len(data)):
                        if len(data[j]) <= i:
                            # This needs to be changed if data length changes
                            writing += [i, 200, 200, 0, 0, None]
                        else:
                            writing += data[j][i] + [None]
                    # writing = list(itertools.chain.from_iterable([[i, 200, 200, 0, None] if len(run) > i else run[i] + [None] for run in data]))
                    if i == 0:
                        writing += ["avg_reward", "avg_avg_reward"]
                    else:
                        # This needs to be changed if data length changes
                        writing += [statistics.mean([float(x) for x in writing[1::6]]), statistics.mean([float(x) for x in writing[2::6]])]
                    csv_writer.writerow(writing)
                    i += 1
            [os.remove(f) for f in files]
        else:
            print("Some hosts are not localhost, not combining files" + worker_hosts)

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
    timer = time.time()

    ps_hosts = FLAGS.ps_hosts.split(",") if FLAGS.ps_hosts else config.get(FLAGS.config, 'ps_hosts').split(",")
    worker_hosts = FLAGS.worker_hosts.split(",") if FLAGS.worker_hosts else config.get(FLAGS.config, 'worker_hosts').split(",")
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
    backup = config.getint(FLAGS.config, 'backup')  # unused in async
    sync = config.getboolean(FLAGS.config, 'sync')
    gradient_prio = False if not sync else config.getboolean(FLAGS.config, 'gradient_prio')
    sync_workers = len(worker_hosts)-backup
    mute = FLAGS.mute if FLAGS.mute else config.getboolean(FLAGS.config, 'mute')
    animate = 0
    draw = 0

    print("Config:\nps_hosts={}\nworker_hosts={}\njob_name={}\ntask_index={}\nlearning_rate={}\n"
          "batch_size={}\nmemory_size={}\ntarget_update={}\nseed={}\ncomm_rounds={}\nepochs={}\n"
          "end_epoch={}\nepoch_decay={}\nnbackup={}\nsync={}"
          .format(ps_hosts, worker_hosts, job, task, learning_rate, batch_size, memory_size, target_update,
                  seed, max_comm_rounds, epochs, end_epoch, epoch_decay, backup, sync))

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    chief = True if job == 'worker' and task == 0 else False
    print("/job:", job, "/task:", task, " - Chief: ", chief, sep='')

    # Create server
    server = tf.train.Server(cluster, job_name=job, task_index=task)

    run_code = "{}-{}-p-{}-w-{}-E-{}-b-{}-m-{}-N-{}-lr-{}-B-{}-s-{}-".\
        format(datetime.now().strftime("%y%m%d-%H%M%S"), env_name, len(ps_hosts), len(worker_hosts),
               epochs, batch_size, memory_size, target_update, learning_rate, backup, seed)
    run_code += "-sync" if sync else "-async"

    # Set a unique random seed for each client
    seed = ((seed * 10) + task)
    random.seed(seed)

    if not mute:
        print("Run code:", run_code)

    # Start parameter servers
    if job == 'ps':
        server.join()

    # Start training
    with U.make_session(num_cpu=4, target=server.target) as sess:
        # Create the environment
        env = gym.make(env_name)
        env.seed(seed)
        tf.set_random_seed(seed)

        # Create all the functions necessary to train the model
        act, train, global_opt,  update_target, update_weights, sync_opt, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            # optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
            chief=chief,
            server=server,
            workers=sync_workers
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(memory_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        if not chief:
            if not mute:
                print("Worker {}/{} will sleep (3s) for chief to initialize variables".format(task+1, len(worker_hosts)))
            time.sleep(4)

        # Initialize the parameters and copy them to the target network.
        U.initialize(chief=chief)

        if chief:
            sess.run(debug['run_code'].assign(run_code))
            if not mute:
                print("Set global run code to:", run_code)

        if not mute:
            print("initialized variables, sleeping for 1 sec")
        time.sleep(2)

        if not chief:
            while not sess.run(tf.is_variable_initialized(debug['run_code'])):
                if not mute:
                    print("Global run code not yet initialized")
                time.sleep(2)
            run_code = str(sess.run(debug['run_code']).decode())
            if run_code == '':
                if not mute:
                    print("Run code empty. Trying to fetch again...")
                time.sleep(5)
            if not mute:
                print("Read global run code:", run_code)

        run_code += "(w" + str(task) + ")"
        print("Final run_code:", run_code)

        t_global_old = update_weights()[0][0]
        update_target()
        exp_gen = 1000  # For how many timesteps sould we only generate experience (not train)
        t_start = exp_gen
        comm_rounds = 0
        comm_rounds_global = 0
        dt = 0
        write_csv(run_code, log=["episode", "reward" + str(task), "avg_reward" + str(task), "t_global", "cr"])
        # TODO RE-ENABLE comm-rounds LOGGING
        # write_csv(run_code, log=["comm_rounds", "t" + str(task), "staleness" + str(task), "epoch" + str(task)],
        #           comm_rounds=True)

        episode_rewards = [0.0]
        cr_reward = 0
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            cr_reward += rew

            # Animate every <animate> episodes
            if not mute and chief and animate > 0 and (len(episode_rewards) % animate) == 0:
                if done:
                    print("ep", len(episode_rewards), "ended with reward:", episode_rewards[-1])
                env.render()

            if done:
                if not mute and chief and draw > 0 and len(episode_rewards) % draw == 0:
                    env.render()
                avg_rew = np.round(np.mean(np.array(episode_rewards[-100:])), 1)
                write_csv(run_code, [len(episode_rewards), episode_rewards[-1], avg_rew, debug['t_global']()[0], comm_rounds_global])

                # Reset and prepare for next episode
                obs = env.reset()
                episode_rewards.append(0)

            [converged] = sync_opt['check_converged']()
            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= max_reward or converged
            if is_solved or comm_rounds >= max_comm_rounds:
                sync_opt['set_converged']([True])
                if not mute:
                    print("Converged was set to", sync_opt['check_converged']()[0])
                write_csv_final(run_code, str(len(episode_rewards)), worker_hosts, chief, comm_rounds_global, mute)
                print("Converged after:  ", len(episode_rewards), "episodes")
                print("Agent total steps:", t)
                print("Global steps:     ", debug['t_global']()[0])
                sec = round(time.time() - timer)
                print("Total time:", sec // 3600, "h", (sec % 3600) // 60, "min", sec % 60, "s")
                return
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t >= exp_gen:
                # if t >= batch_size:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    if t - t_start >= np.round(epoch.value(comm_rounds)):  # The number of local timesteps to calculate before averaging gradients
                        # print("t = {}, Updating global network (t_global = {})".format(t, debug['t_global']()[0]))

                        cr_old = comm_rounds_global

                        # Apply gradients to weights in PS
                        if sync:
                            # Tell the ps we are done and want to submit score
                            [[comm_rounds_global], [worker_count]] = sync_opt['request_submit']()
                            # print("Checking if CRG(", comm_rounds_global, ") = CR(", comm_rounds, ")", sep="")

                            if comm_rounds_global == comm_rounds:
                                # print("Checking if WC(", worker_count, ") <= n(", sync_workers, ")", sep="")
                                if worker_count <= sync_workers:
                                    # If allowed to submit score, do it
                                    [comm_rounds_global] = sync_opt['submit_score']([cr_reward])

                                    # print("WC Safe:", worker_count, "<=", sync_workers)

                                    if chief: #worker_count == sync_workers:
                                        # print("Chief done: round", comm_rounds_global)
                                        # If it is the last submission of the round, start gradient update round
                                        [submits] = sync_opt['set_submit']([0])
                                        while worker_count != sync_workers:
                                            if sync_opt['check_converged']()[0]:
                                                if not mute:
                                                    print("Other worker converged! Finishing in check_wc")
                                                break
                                            worker_count = sync_opt['check_wc']()[0]

                                    while sync_opt['check_submit']()[0] == -1:
                                        if sync_opt['check_converged']()[0]:
                                            if not mute:
                                                print("Other worker converged! Finishing in check_submit")
                                            break
                                        # print("Waiting for submit to start")
                                        # time.sleep(0.1)
                                        pass

                                    if sync_opt['check_converged']()[0]:
                                        if not mute:
                                            print("Other worker converged! Continuing before submit")
                                        continue

                                    # Now all eligible workers have sent their score and gradient round has started
                                    # Submit gradient
                                    # TODO 4th argument overrides everything else unles it is set to -1 in the code
                                    [[dt], [comm_rounds_global], [factor]] = global_opt([t - t_start], [t_global_old],
                                                                              [cr_reward], [1/len(worker_hosts)], [True])

                                    submits = sync_opt['inc_submit']()
                                    # print("Score=", cr_reward, " Submits=", submits[0][0], " t=", t-t_start,
                                    #       " t_global_old=", t_global_old, " cr_old=", cr_old,
                                    #       " cr_global=", comm_rounds_global, " dt=", dt, " factor=", factor, sep='')

                                    # Chief waits until submits = n
                                    if chief:
                                        while not sync_opt['check_submit']()[0] == sync_workers:
                                            if sync_opt['check_converged']()[0]:
                                                if not mute:
                                                    print("Other worker converged! Finishing in check_submit (chief)")
                                                break
                                            # print("Chief waiting for all submits", sync_opt['check_submit']()[0], "!=", sync_workers)
                                            #time.sleep(5)
                                            pass
                                        # print("Round", comm_rounds, "finished")
                                        [w] = sync_opt['reset_wc']()[0]
                                        # print("Worker count reset to:", w)
                                        sync_opt['reset_score']()
                                        submits = sync_opt['set_submit']([-1])
                                        # print("Submit round finished. Submits set to:", submits[0])
                                        [r] = sync_opt['inc_comm_round']()[0]
                                        # print("New round started:", r)

                                    # Normal workers wait until GCR > CR
                                    if not chief:
                                        while sync_opt['check_round']()[0] <= comm_rounds:
                                            if sync_opt['check_converged']()[0]:
                                                if not mute:
                                                    print("Other worker converged! Finishing in check_round")
                                                break
                                            # print("Worker submitted, waiting for next round:", comm_rounds + 1)
                                            # time.sleep(0.1)
                                            pass

                                else: #elif worker_count > sync_workers:
                                    # If not allowed to submit score, wait for next round to start
                                    if not mute:
                                        print("Worker finished too late but before new round started (", comm_rounds_global, ")")
                                        print("WC(", worker_count, ") > N(", sync_workers, ")", sep="")
                                    target = np.floor(comm_rounds_global + 1)  # +1 if x.0, +0.5 if x.5
                                    while not sync_opt['check_round']()[0] >= target:
                                        pass

                            elif comm_rounds_global > comm_rounds:
                                # This means the worker is behind. Do nothing and start next round
                                if not mute:
                                    print("Communication round ", comm_rounds, "missed. Actual round:", comm_rounds_global)
                                # TODO How to handle round count when skipping rounds?
                                comm_rounds = comm_rounds_global - 1

                            elif comm_rounds_global < comm_rounds:
                                print("WARNING! Worker ahead of global:", comm_rounds, ">", comm_rounds_global)
                                time.sleep(5)

                        else:
                            sync_opt['inc_comm_round']()
                            [[dt], [comm_rounds_global], [factor]] = global_opt([t - t_start], [t_global_old], [0], [-1], [False])

                        # Update the local weights with the new global weights from PS
                        t_global_old = update_weights()[0][0]

                        comm_rounds += 1
                        # print("Round finished. Increasing local comm_round to:", comm_rounds)
                        cr_reward = 0
                        # TODO RE-ENABLE comm-rounds LOGGING
                        # write_csv(run_code, [comm_rounds, t, dt, epoch.value(comm_rounds)], comm_rounds=True)

                        t_start = t
                        # epochs = end_epoch if epochs <= end_epoch else epochs - epoch_decay_rate

                # Update target network periodically.
                if t % target_update == 0:
                    update_target()

            if not mute and done and len(episode_rewards) % 10 == 0:
                last_rewards = episode_rewards[-101:-1]
                logger.record_tabular("steps", t)
                logger.record_tabular("global steps", debug['t_global']()[0])
                logger.record_tabular("communication rounds", comm_rounds)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", np.round(np.mean(episode_rewards[-101:-1]), 4))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger.record_tabular("last gradient factor", np.round(factor, 4))
                logger.dump_tabular()
                rew_ill = ['●' if x >= max_reward else str(int(np.floor(x / (max_reward/10)))) if x >= (max_reward/10) else '_' for x in last_rewards]
                streak = 0
                for i in reversed(rew_ill):
                    if i == "●":
                        streak += 1
                    else:
                        break
                print("[" + ''.join(rew_ill) + "] ([● " + str(rew_ill.count('●')) + " | " + str(rew_ill.count('9')) +
                      " | " + str(rew_ill.count('8')) + " | " + str(rew_ill.count('7')) +
                      " | " + str(rew_ill.count('6')) + " | " + str(rew_ill.count('5')) +
                      " | " + str(rew_ill.count('4')) + " | " + str(rew_ill.count('3')) +
                      " | " + str(rew_ill.count('2')) + " | " + str(rew_ill.count('1')) +
                      " | " + str(rew_ill.count('_')) + " _]/" + str(len(rew_ill)) + " {S:" + str(streak) + "})", sep='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
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
    parser.add_argument(
        "--mute",
        type=bool,
        default=False,
        help="Reduce spam"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("Unparsed:", unparsed)
    tf.app.run(main=main)
