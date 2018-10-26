from copy import deepcopy
import argparse

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm
from quadprog import solve_qp


def is_pd(K):
    """
    Checks if matrix is psd
    """
    try:
        np.linalg.cholesky(K)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0
        else:
            raise


def compute_trajectory(env, k=None, K=None, alpha=1, old_xs=None, old_us=None):

    # generating ranom trajectory
    steps = 0
    done = False
    scores = []
    new_xs = []
    new_us = []

    x = env.reset()
    while not done:

        # get random action
        if k is None or K is None:
            u = np.random.uniform(-max_action, max_action, action_dim)

        # compute new action
        else:
            u = old_us[steps] + alpha * k[steps] + \
                K[steps] @ (old_xs[steps] - x)

        # get state
        x_n, r, done, _ = env.step(u)

        # update stuff
        new_us.append(u)
        new_xs.append(x)
        scores.append(r)
        steps += 1
        x = x_n

        # reset when done
        if done:
            env.reset()

    return new_xs, new_us, scores, steps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')
    args = parser.parse_args()

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    ###### iLQG Algorithm ######

    n_steps = 1000
    total_steps = 0
    action_dim = action_dim
    state_dim = state_dim
    max_action = max_action

    # open-loop term k and feedback gain K
    k = np.zeros(n_steps, action_dim)
    K = np.zeros(n_steps, action_dim, state_dim)

    # derivatives
    f_x = np.zeros(n_steps, state_dim, state_dim)
    f_u = np.zeros(n_steps, state_dim, action_dim)
    l_x = np.zeros(n_steps, state_dim)
    l_u = np.zeros(n_steps, action_dim)
    l_uu = np.zeros(n_steps, action_dim, action_dim)
    l_xx = np.zeros(n_steps, state_dim, state_dim)

    # quadratic approx terms
    q_x = np.zeros(n_steps, state_dim)
    q_u = np.zeros(n_steps, action_dim)
    q_uu = np.zeros(n_steps, action_dim, action_dim)
    q_ux = np.zeros(n_steps, action_dim, state_dim)
    q_xx = np.zeros(n_steps, state_dim, state_dim)

    # state values
    v_x = np.zeros(n_steps, state_dim)
    v_xx = np.zeros(n_steps, state_dim, state_dim)

    # hyper-parameters
    alphas = 10. ^ np.linspace(0, -3, 11)
    mu = 1
    d_mu = 1
    mu_f = 1.6
    mu_max = 1e10
    mu_min = 1e-6
    tol = 10**-7
    z_min = 0

    # generating ranom trajectory
    xs, us, scores, steps = compute_trajectory(env)

    # main loop
    alg_done = False
    while not alg_done:

        # compute derivatives

        # backward pass this should be done
        backward_failed = True
        while backward_failed:

            for i in range(n_steps-2, -1, -1):

                # computing q values
                q_x[i] = l_x[i] + f_x[i].T @ v_x[i+1]
                q_u[i] = l_u[i] + f_u[i].T @ v_x[i+1]
                q_uu[i] = l_uu[i] + \
                    f_u[i].T @ (v_xx[i+1] + mu * np.ones(state_dim)) @ f_u[i]
                q_ux[i] = l_uu[i] + \
                    f_u[i].T @ (v_xx[i+1] + mu * np.ones(state_dim)) @ f_x[i]
                q_xx[i] = l_xx[i] + \
                    f_x[i].T @ (v_xx[i+1] + mu * np.ones(state_dim)) @ f_x[i]

                # checking if positive definite
                if not is_pd(q_uu[i]):
                    backward_failed = True
                    break
                else:
                    backward_failed = False
                    q_uu_inv = np.linalg.inv(q_uu[i])

                # solving quadratic problem to get new actions
                G = q_uu[i]
                a = -q_u[i]
                C = np.concatenate(
                    [np.eye(action_dim), -np.eye(action_dim)], axis=0)
                lower = (-1 - us[i]) * np.ones(action_dim)
                upper = (us[i] - 1) * np.ones(action_dim)
                b = np.concatenate([lower, upper], axis=0)
                k_i, f, k_u, iters, lagr, iact = solve_qp(G, a, C, b)

                # updating open loop term and feedback gain
                k[i] = k_i
                K[i] = np.zeros(action_dim, state_dim)
                for j in range(action_dim):
                    if j not in iact:
                        K[j] = -q_uu_inv @ q_ux[j]

                # updating cost to go approximations
                dV = 1/2 * k[i].T @ q_uu[i] @ k[i]
                dV += k[i].T @ q_u[i]
                v_x[i] = q_x[i] + K[i].T @ q_uu[i] @ k[i]
                v_x[i] += K[i].T @ q_u[i] + q_ux[i].T@ k[i]
                v_xx[i] = q_xx[i] + K[i].T @ q_uu[i]@ K[i]
                v_xx[i] += K[i].T@ q_ux[i] + q_ux[i].T @ K[i]

            if backward_failed:
                print("Non-psd matrix encountered, increasing mu")
                d_mu = max(d_mu * mu_f, mu_f)
                mu = max(mu * d_mu, mu_min)

        # forward pass
        for alpha in alphas:

            forward_failed = False
            new_xs, new_us, new_scores, steps = compute_trajectory(
                env, k, K, alpha, xs, us)
            d_score = np.sum(scores) - np.sum(new_scores)
            d_score_e = -alpha * (dV[0] + alpha * dV[1])

            if d_score_e > 0:
                z = d_score / d_score_e

            else:
                z = np.abs(d_score - d_score_e)
                print("Non-positive reward reduction")

            if z > z_min:
                print("Forward pass success")
                break

            forward_failed = True

        # updating states and trajectories
        if not forward_failed:

            xs = new_xs
            us = new_us
            scores = new_scores
            total_steps += steps

            d_mu = min(d_mu / mu_f, 1. / mu_f)
            mu = mu * d_mu * (mu > mu_min)

            # are we done ?
            if d_score < tol:
                alg_done = True

        else:
            d_mu = max(d_mu * mu_f, mu_f)
            mu = max(mu * d_mu, mu_min)

            if mu > mu_max:
                print("mu > mu_max")
                alg_done = True
