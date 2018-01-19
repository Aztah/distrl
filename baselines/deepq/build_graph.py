"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = hauber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import baselines.common.tf_util as U


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, task=0):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        with tf.device("/job:worker/task:{}".format(task)):
            observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
            stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

            eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

            q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
            deterministic_actions = tf.argmax(q_values, axis=1)

            batch_size = tf.shape(observations_ph.get())[0]
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

            output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
            update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

            act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                             outputs=output_actions,
                             givens={update_eps_ph: -1.0, stochastic_ph: True},
                             updates=[update_eps_expr])
            return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, chief=False, server=None, workers=1,
                grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    chief: bool
        whether or not the worker should assume chief duties.
        these include: initializing global parameters, tensorboarding, saving, etc.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    task = server.server_def.task_index
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse, task=task)

    with tf.variable_scope(scope, reuse=reuse):
        with tf.device("/job:worker/task:{}".format(task)):
            # set up placeholders
            obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
            act_t_ph = tf.placeholder(tf.int32, [None], name="action")
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
            obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
            importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

            # Local timestep counters
            t = tf.placeholder(tf.float32, [1], name="t")
            t_global_old = tf.placeholder(tf.float32, [1], name="t_global_old")
            score_input = tf.placeholder(tf.float32, [1], name="score_input")
            grad_prio = tf.placeholder(tf.bool, [1], name="grad_prio")
            converged_ph = tf.placeholder(tf.bool, [1], name="converged")
            factor_input = tf.placeholder(tf.float32, [1], name="factor_input")

            # Global timestep counter
            # TODO Does TF have built-in global step counters?
            with tf.device("/job:ps/task:0"):
                t_global = tf.Variable(dtype=tf.float32, initial_value=[0], name="t_global")
                run_code_global = tf.Variable(initial_value="", name="run_code_global")
                comm_rounds_global = tf.Variable(dtype=tf.float32, initial_value=[0], name="comm_rounds_global")
                max_workers_global = tf.constant(workers, dtype=tf.float32, name="max_workers_global")
                worker_count_global = tf.Variable(dtype=tf.float32, initial_value=[0], name="worker_count_global")
                score_max_global = tf.Variable(dtype=tf.float32, initial_value=[0], name="score_max_global")
                score_min_global = tf.Variable(dtype=tf.float32, initial_value=[0], name="score_min_global")
                submit_count_global = tf.Variable(dtype=tf.float32, initial_value=[-1], name="submit_count_global")
                converged_global = tf.Variable(dtype=tf.bool, initial_value=[False], name="converged_global")

            # q network evaluation
            q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
            q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

            # target q network evalution
            q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
            target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

            # global weights
            print("chief:", chief, "reuse:", True if not chief else None)
            global_q_func_vars = []
            # with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            with tf.device("/job:ps/task:0"):  # TODO needs RDS if using multiple PS
                # q_global = q_func(obs_t_input.get(), num_actions, scope="global_weights", reuse=None if chief else True)#reuse=(not chief))
                # q_global = q_func(obs_t_input.get(), num_actions, scope="global_weights")
                with tf.variable_scope("global_weights"):
                    for var in q_func_vars:
                        name = var.name.split(":")[0].split("q_func/")[-1]
                        global_q_func_vars.append(
                            tf.get_variable(name=name, shape=var.shape, dtype=var.dtype,
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1, dtype=var.dtype)))
            # global_q_func_vars = U.scope_vars(U.absolute_scope_name("global_weights"))
            # print("Global:", global_q_func_vars)

            # old weights (used to implicitly calculate gradient sum: q_func_vars - q_func_vars_old)
            q_func_vars_old = []
            with tf.variable_scope("old_weights"):
                for var in q_func_vars:
                    name = var.name.split(":")[0].split("q_func/")[-1]
                    q_func_vars_old.append(
                        tf.get_variable(name=name, shape=var.shape, dtype=var.dtype,
                                        initializer=tf.contrib.layers.xavier_initializer(seed=1, dtype=var.dtype)))
            # q_old = q_func(obs_t_input.get(), num_actions, scope="old_weights")
            # q_func_vars_old = U.scope_vars(U.absolute_scope_name("old_weights"))
            # print("Old vars:", q_func_vars_old)

            # q scores for actions which we know were selected in the given state.
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

            # compute estimate of best possible value starting from state at t + 1
            if double_q:
                q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
                q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)
            q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

            # compute the error (potentially clipped)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = U.huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights_ph * errors)

            # compute optimization op (potentially with gradient clipping)
            if grad_norm_clipping is not None:
                optimize_expr = U.minimize_and_clip(optimizer,
                                                    weighted_error,
                                                    var_list=q_func_vars,
                                                    clip_val=grad_norm_clipping)
            else:
                optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_expr = []
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            update_target_expr = tf.group(*update_target_expr)

            # update_global_fn will be called periodically to copy global Q network to q network
            update_global_expr = []
            for var_global, var, var_old in zip(sorted(global_q_func_vars, key=lambda v: v.name),
                                                sorted(q_func_vars, key=lambda v: v.name),
                                                sorted(q_func_vars_old, key=lambda v: v.name)):
                update_global_expr.append(var.assign(var_global))
                # TODO Can async cause var <- var_global, var_global <- new value, var_old <- var_global in that order?
                # TODO Should this copy from var instead? (concurrency issues?)
                # TODO Can concurrency cause var_old <- var, var <- var_global in that order (resulting in wrong values)?
                # TODO Safest method is to force sequential execution of var <- var_global, var_old <- var! How though?
                update_global_expr.append(var_old.assign(var_global))
            update_global_expr = tf.group(*update_global_expr)

            # update the global time step counter by adding the local
            update_t_global = t_global.assign_add(t)

            optimize_global_expr = []
            # Factor to multiply every gradient with
            # f = t / (t_global - t_global_old)
            dt = tf.subtract(update_t_global, t_global_old)
            factor = tf.where(tf.greater_equal(factor_input, 0),
                              factor_input,
                              tf.where(grad_prio,
                              tf.divide(tf.subtract(score_input, score_min_global),
                                        tf.subtract(score_max_global, score_min_global)),
                              tf.div(t, dt)))
            for var, var_old, var_global in zip(sorted(q_func_vars, key=lambda v: v.name),
                                                sorted(q_func_vars_old, key=lambda v: v.name),
                                                sorted(global_q_func_vars, key=lambda v: v.name)):
                # Multiply the difference between the old parameters and the locally optimized parameters
                # g = (var - var_old) * f
                grad = tf.multiply(tf.subtract(var, var_old), factor)
                optimize_global_expr.append(var_global.assign_add(grad))
            optimize_global_expr = tf.group(*optimize_global_expr)

            # if cr == cr_g and wc < wc_max:
            #   wc += 1
            #   score_global += score
            # if cr == cr_g and wc == wc_max:
            #   vc += 1
            #   score_global += score
            #   cr_g += 0.5
            # return cr_g
            """
            if cr == cr_g:
                if wc <= wc_max:
                    wc += 1
                    score_global += score
                    if wc == wc_max:
                        cr_g += 0.5
            return cr_g
            """
            # submit_score_expr = \
            #     tf.cond(tf.equal(comm_rounds, comm_rounds_global),
            #             lambda: tf.cond(tf.less_equal(worker_count_global, max_workers_global),
            #                             lambda: tf.group(worker_count_global.assign_add([1]),
            #                                              score_global.assign_add(score_input),
            #                                              tf.cond(tf.equal(worker_count_global, max_workers_global),
            #                                                      lambda: comm_rounds_global.assign_add([0.5]),
            #                                                      lambda: None)),
            #                             lambda: tf.group(None, None, None)),
            #             lambda: None)
            # submit_score_expr = \
            #     tf.cond(tf.logical_and(tf.equal(comm_rounds, comm_rounds_global),
            #                            tf.less(worker_count_global, max_workers_global)),
            #             tf.group(worker_count_global.assign_add(1),
            #                      score_global.assign_add(score_input)),
            #             tf.cond(tf.logical_and(tf.equal(comm_rounds, comm_rounds_global),
            #                                    tf.equal(worker_count_global, max_workers_global)),
            #                     tf.group(worker_count_global.assign_add(1),
            #                              score_global.assign_add(score_input),
            #                              comm_rounds_global.assign_add(0.5))))

            # This makes a sum of all scores (
            # submit_score_expr = score_global.assign_add(score_input)

            # This only saves the maximum score (for normalized score weighting)
            submit_score_max = score_max_global.assign(tf.maximum(score_input, score_max_global), use_locking=True)
            submit_score_min = score_min_global.assign(tf.minimum(score_input, score_min_global), use_locking=True)

            set_submit_count = submit_count_global.assign(score_input, use_locking=True)
            inc_submit_count = submit_count_global.assign_add([1], use_locking=True)

            # check_round_op = tf.equal(comm_rounds, comm_rounds_global) # Not used anymore
            inc_wc = worker_count_global.assign_add([1], use_locking=True)
            zero_wc = worker_count_global.assign([0], use_locking=True)

            inc_cr = comm_rounds_global.assign_add([1], use_locking=True)

            score_reset = score_max_global.assign([0], use_locking=True)

            converged_set = converged_global.assign(converged_ph, use_locking=True)


            # Create callable functions
            train = U.function(
                inputs=[
                    obs_t_input,
                    act_t_ph,
                    rew_t_ph,
                    obs_tp1_input,
                    done_mask_ph,
                    importance_weights_ph
                ],
                outputs=[td_error],
                updates=[optimize_expr]
            )
            global_opt = U.function(inputs=[t, t_global_old, score_input, factor_input, grad_prio], outputs=[dt, comm_rounds_global, factor], updates=[optimize_global_expr])
            # global_sync_opt = U.function(inputs=[comm_rounds], outputs=[comm_rounds_global], updates=[optimize_global_sync_expr])
            update_weights = U.function(inputs=[], outputs=[t_global], updates=[update_global_expr])
            update_target = U.function([], [], updates=[update_target_expr])
            submit_score = U.function(inputs=[score_input], outputs=[comm_rounds_global], updates=[submit_score_max, submit_score_min])
            check_round = U.function(inputs=[], outputs=[comm_rounds_global], updates=[])
            request_submit = U.function(inputs=[], outputs=[comm_rounds_global, inc_wc], updates=[])
            set_submit = U.function(inputs=[score_input], outputs=[set_submit_count], updates=[])
            check_submit = U.function(inputs=[], outputs=[submit_count_global], updates=[])
            inc_submit = U.function(inputs=[], outputs=[inc_submit_count], updates=[])
            inc_comm_round = U.function(inputs=[], outputs=[inc_cr], updates=[])
            reset_wc = U.function(inputs=[], outputs=[zero_wc], updates=[])
            check_wc = U.function(inputs=[], outputs=[worker_count_global], updates=[])
            reset_score = U.function(inputs=[], outputs=[], updates=[score_reset])
            set_converged = U.function(inputs=[converged_ph], outputs=[], updates=[converged_set])
            check_converged = U.function(inputs=[], outputs=[converged_global], updates=[])

            # Debugging functions
            q_values = U.function([obs_t_input], q_t)
            weights = U.function(inputs=[], outputs=[q_func_vars, global_q_func_vars, q_func_vars_old], updates=[])
            t_global_func = U.function([], t_global)
            comm_rounds_func = U.function([], comm_rounds_global)

            return act_f, train, global_opt, update_target, update_weights, \
                {'request_submit': request_submit, 'submit_score': submit_score,
                 'check_round': check_round, 'check_submit': check_submit, 'set_submit': set_submit,
                 'inc_submit': inc_submit, 'inc_comm_round': inc_comm_round, 'reset_wc': reset_wc,
                 'check_wc': check_wc, 'reset_score': reset_score,
                 'set_converged': set_converged, 'check_converged': check_converged}, \
                {'q_values': q_values, 'weights': weights, 't_global': t_global_func,
                 'run_code': run_code_global, 'comm_rounds': comm_rounds_func, 'factor': factor}

