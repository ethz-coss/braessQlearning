import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats
import scipy.cluster
from scipy.spatial.distance import pdist


def count_groups(Q_values, dist):
    y = scipy.cluster.hierarchy.average(Q_values)
    z = scipy.cluster.hierarchy.fcluster(y, dist, criterion='distance')
    groups = np.bincount(z)
    return len(groups)


def vecSOrun(N_AGENTS, N_ACTIONS, N_ITER, EPSILON, mask, GAMMA, ALPHA, QINIT, PAYOFF_TYPE, PAYOFF_NOISE):
    if type(QINIT) == np.ndarray:
        Q = QINIT.T * np.ones((N_AGENTS, N_ACTIONS))

    elif QINIT == "UNIFORM":
        Q = - np.random.random_sample(size=(N_AGENTS, N_ACTIONS)) - 1

    if ALPHA == "UNIFORM":
        ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    M = {}

    for t in range(N_ITER):

        ## DETERMINE ACTIONS
        rand = np.random.random_sample(size=N_AGENTS)
        # print(rand)
        randA = np.random.randint(N_ACTIONS, size=N_AGENTS)
        # print(randA)
        A = np.where(rand >= EPSILON,
                     np.argmax(Q, axis=1),
                     randA)
        ## DETERMINE PAYOFFS PER PLAYER
        if N_ACTIONS == 2:
            mean = np.mean(A)
            r_0 = 2 - mean
            r_1 = 1 + mean
            T = [-r_0, -r_1]

            R = -1 * np.where(A == 0, r_0, r_1)

        elif N_ACTIONS == 3:
            n_up = (A == 0).sum()
            n_down = (A == 1).sum()
            n_cross = (A == 2).sum()

            r_0 = 1 + (n_up + n_cross) / N_AGENTS
            r_1 = 1 + (n_down + n_cross) / N_AGENTS
            r_2 = (n_up + n_cross) / N_AGENTS + (n_down + n_cross) / N_AGENTS
            T = [-r_0, -r_1, -r_2]

            dict_map = {0: r_0, 1: r_1, 2: r_2}

            R = -1 * np.vectorize(dict_map.get)(A)

        if PAYOFF_TYPE == "SOCIAL":
            W = welfare(R, N_AGENTS)
            R = W * np.ones(N_AGENTS)

        if PAYOFF_NOISE == "GAUSSIAN":
            R += np.random.normal(0, 0.05, size=N_AGENTS)

        ## UPDATE AGENT Q VALUES
        ind = np.arange(N_AGENTS)
        Q[ind, A] = Q[ind, A] + ALPHA * (R + GAMMA * Q.max(axis=1) - Q[ind, A])
        Qmean = Q.mean(axis=0)

        ## SAVE PROGRESS DATA
        # M[t] = {"A": A, "R": R}
        M[t] = {"A": A, "nA": np.bincount(A, minlength=3), "R": R, "T": T,
                "Qmean": Qmean}

    return M, Q


def vecDisruptedRun(N_AGENTS, N_ACTIONS, N_ITER, EPSILON, mask, GAMMA, ALPHA, QINIT, PAYOFF_TYPE, PAYOFF_NOISE, DISRUPTIONS):
    if type(QINIT) == np.ndarray:
        Q = QINIT.T * np.ones((N_AGENTS, N_ACTIONS))

    elif QINIT == "UNIFORM":
        Q = - np.random.random_sample(size=(N_AGENTS, N_ACTIONS)) - 1

    if ALPHA == "UNIFORM":
        ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    M = {}
    disruptionCounter = 0

    for t in range(N_ITER):

        ## DETERMINE ACTIONS
        rand = np.random.random_sample(size=N_AGENTS)
        # print(rand)
        randA = np.random.randint(N_ACTIONS, size=N_AGENTS)
        # print(randA)
        A = np.where(rand >= EPSILON,
                     np.argmax(Q, axis=1),
                     randA)

        if t in DISRUPTIONS:
            disruptionCounter += 1

        if disruptionCounter > 0:
            A[0:int(1 / 3 * N_AGENTS)] = 2
            disruptionCounter += 1

        if disruptionCounter == 501:
            disruptionCounter = 0

        ## DETERMINE PAYOFFS PER PLAYER
        if N_ACTIONS == 2:
            mean = np.mean(A)
            r_0 = 2 - mean
            r_1 = 1 + mean
            T = [-r_0, -r_1]

            R = -1 * np.where(A == 0, r_0, r_1)

        elif N_ACTIONS == 3:
            n_up = (A == 0).sum()
            n_down = (A == 1).sum()
            n_cross = (A == 2).sum()

            r_0 = 1 + (n_up + n_cross) / N_AGENTS
            r_1 = 1 + (n_down + n_cross) / N_AGENTS
            r_2 = (n_up + n_cross) / N_AGENTS + (n_down + n_cross) / N_AGENTS
            T = [-r_0, -r_1, -r_2]

            dict_map = {0: r_0, 1: r_1, 2: r_2}

            R = -1 * np.vectorize(dict_map.get)(A)

        if PAYOFF_TYPE == "SOCIAL":
            W = welfare(R, N_AGENTS)
            R = W * np.ones(N_AGENTS)

        if PAYOFF_NOISE == "GAUSSIAN":
            R += np.random.normal(0, 0.1, size=N_AGENTS)

        ## UPDATE AGENT Q VALUES
        ind = np.arange(N_AGENTS)
        Q[ind, A] = Q[ind, A] + ALPHA * (R + GAMMA * Q.max(axis=1) - Q[ind, A])
        Qmean = Q.mean(axis=0)

        ## SAVE PROGRESS DATA
        # M[t] = {"A": A, "R": R}
        M[t] = {"A": A, "nA": np.bincount(A, minlength=3), "R": R, "T": T,
                "Qmean": Qmean}

    return M, Q


def next_state(A):
    A = np.pad(A, (1, 1), "wrap")
    N = sliding_window_view(A, window_shape=3)
    left_collision = np.equal(N[:, 0], N[:, 1])
    right_collision = np.equal(N[:, 2], N[:, 1])
    S = left_collision * 1 + right_collision * 1  # multiplication by one gets, bool --> int

    return S


def vecSOrun_states(N_AGENTS, N_STATES, N_ACTIONS, NEIGHBOURS, N_ITER, EPSILON, mask, GAMMA, ALPHA, QINIT, PAYOFF_TYPE, SELECT_TYPE):
    if type(QINIT) == np.ndarray:
        if QINIT.shape == (N_AGENTS, N_STATES, N_ACTIONS):
            Q = QINIT
        else:
            Q = QINIT.T * np.ones((N_AGENTS, N_STATES, N_ACTIONS))
    elif QINIT == "UNIFORM":
        Q = - np.random.random_sample(size=(N_AGENTS, N_STATES, N_ACTIONS)) - 1

    if ALPHA == "UNIFORM":
        ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    M = {}

    S = np.random.randint(N_STATES, size=N_AGENTS)

    indices = np.arange(N_AGENTS)

    for t in range(N_ITER):

        if SELECT_TYPE == "EPSILON":
            ## DETERMINE ACTIONS
            rand = np.random.random_sample(size=(N_AGENTS))
            # print(rand)
            randA = np.random.randint(N_ACTIONS, size=N_AGENTS)
            # print(randA)
            A = np.where(rand >= EPSILON,
                         np.argmax(Q[indices, S, :], axis=1),
                         randA)
        elif SELECT_TYPE == "gnet":
            pass

        ## DETERMINE NEXT STATE
        if N_STATES > 1:
            S = next_state(A)

        ## DETERMINE PAYOFFS PER PLAYER
        if N_ACTIONS == 2:
            mean = np.mean(A)
            r_0 = 2 - mean
            r_1 = 1 + mean
            T = [-r_0, -r_1]

            R = -1 * np.where(A == 0, r_0, r_1)

        elif N_ACTIONS == 3:
            n_up = (A == 0).sum()
            n_down = (A == 1).sum()
            n_cross = (A == 2).sum()

            r_0 = 1 + (n_up + n_cross) / N_AGENTS
            r_1 = 1 + (n_down + n_cross) / N_AGENTS
            r_2 = (n_up + n_cross) / N_AGENTS + (n_down + n_cross) / N_AGENTS
            T = [-r_0, -r_1, -r_2]

            dict_map = {0: r_0, 1: r_1, 2: r_2}

            R = -1 * np.vectorize(dict_map.get)(A)

        if PAYOFF_TYPE == "SOCIAL":
            W = welfare(R, N_AGENTS)
            R = W * np.ones(N_AGENTS)

        if PAYOFF_TYPE == "LOCAL":
            tempR = np.pad(R, (int((NEIGHBOURS - 1)/2), int((NEIGHBOURS - 1)/2)), "wrap")
            N = sliding_window_view(tempR, window_shape=NEIGHBOURS)
            R = np.mean(N, axis=1)

        ## UPDATE AGENT Q VALUES
        ind = np.arange(N_AGENTS)
        Q[ind, S, A] = Q[ind, S, A] + ALPHA * (R + GAMMA * Q[ind, S].max(axis=1) - Q[ind, S, A])
        Qmean = Q.mean(axis=1).mean(axis=0)

        ## SAVE PROGRESS DATA
        # M[t] = {"A": A, "R": R}
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "T": T,
                "Qmean": Qmean,
                "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0)}

    return M, Q


def vecAgentGlobal(N_AGENTS, N_STATES, N_ACTIONS, NEIGHBOURS, N_ITER, EPSILON, mask, GAMMA, ALPHA, QINIT, PAYOFF_TYPE, SELECT_TYPE):
    if type(QINIT) == np.ndarray:
        if QINIT.shape == (N_AGENTS, N_STATES, N_ACTIONS):
            Q = QINIT
        else:
            Q = QINIT.T * np.ones((N_AGENTS, N_STATES, N_ACTIONS))
    elif QINIT == "UNIFORM":
        Q = - np.random.random_sample(size=(N_AGENTS, N_STATES, N_ACTIONS)) - 1

    # Qg = -np.random.random_sample(size=)

    if ALPHA == "UNIFORM":
        ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    M = {}

    S = np.random.randint(N_STATES, size=N_AGENTS)

    indices = np.arange(N_AGENTS)

    for t in range(N_ITER):

        if SELECT_TYPE == "EPSILON":
            ## DETERMINE ACTIONS
            rand = np.random.random_sample(size=(N_AGENTS))
            # print(rand)
            randA = np.random.randint(N_ACTIONS, size=N_AGENTS)
            # print(randA)
            A = np.where(rand >= EPSILON,
                         np.argmax(Q[indices, S, :], axis=1),
                         randA)
        elif SELECT_TYPE == "gnet":
            pass

        ## DETERMINE NEXT STATE
        if N_STATES > 1:
            S = next_state(A)

        ## DETERMINE PAYOFFS PER PLAYER
        if N_ACTIONS == 2:
            mean = np.mean(A)
            r_0 = 2 - mean
            r_1 = 1 + mean
            T = [-r_0, -r_1]

            R = -1 * np.where(A == 0, r_0, r_1)

        elif N_ACTIONS == 3:
            n_up = (A == 0).sum()
            n_down = (A == 1).sum()
            n_cross = (A == 2).sum()

            r_0 = 1 + (n_up + n_cross) / N_AGENTS
            r_1 = 1 + (n_down + n_cross) / N_AGENTS
            r_2 = (n_up + n_cross) / N_AGENTS + (n_down + n_cross) / N_AGENTS
            T = [-r_0, -r_1, -r_2]

            dict_map = {0: r_0, 1: r_1, 2: r_2}

            R = -1 * np.vectorize(dict_map.get)(A)

        if PAYOFF_TYPE == "SOCIAL":
            W = welfare(R, N_AGENTS)
            R = W * np.ones(N_AGENTS)

        if PAYOFF_TYPE == "LOCAL":
            tempR = np.pad(R, (int((NEIGHBOURS - 1)/2), int((NEIGHBOURS - 1)/2)), "wrap")
            N = sliding_window_view(tempR, window_shape=NEIGHBOURS)
            R = np.mean(N, axis=1)

        ## UPDATE AGENT Q VALUES
        ind = np.arange(N_AGENTS)
        Q[ind, S, A] = Q[ind, S, A] + ALPHA * (R + GAMMA * Q[ind, S].max(axis=1) - Q[ind, S, A])
        Qmean = Q.mean(axis=1).mean(axis=0)

        ## SAVE PROGRESS DATA
        # M[t] = {"A": A, "R": R}
        M[t] = {"A": A, "nA": np.bincount(A, minlength=3), "R": R, "T": T,
                "Qmean": Qmean}

    return M, Q


def welfare(R, N_AGENTS, welfareType="AVERAGE"):
    if welfareType == "AVERAGE":
        return R.sum() / N_AGENTS
    elif welfareType == "MIN":
        return R.min()
    elif welfareType == "MAX":
        return R.max()
    else:
        raise "SPECIFY WELFARE TYPE"


def plot_run(M, NAME, N_AGENTS, N_ACTIONS, N_ITER):
    ## PLOTTING EVOLUTION
    print(NAME)

    cmap = plt.get_cmap('plasma')
    a_labels = ["up", "down", "cross"] if N_ACTIONS == 3 else ["up", "down"]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    W = [welfare(M[t]["R"], N_AGENTS, "AVERAGE") for t in range(N_ITER)]

    ax[0, 0].plot(W, color=cmap(0.5))
    ax[0, 0].set_ylim((-2, -1))
    ax[0, 0].set_xlabel('t')
    ax[0, 0].set_ylabel('welfare')
    ax[0, 0].set_title("Average Travel Time")

    x_vals = np.arange(0, N_ITER)
    T = {}
    for a in range(N_ACTIONS):
        T[a] = [M[t]["T"][a] for t in range(N_ITER)]

    ax[0, 1].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, N_ACTIONS)])

    for a in range(N_ACTIONS):
        ax[0, 1].scatter(x_vals, T[a], label=a_labels[a], alpha=0.4)
    ax[0, 1].set_ylim((-2, -1))
    ax[0, 1].set_xlabel('t')
    ax[0, 1].set_ylabel('travel time')
    ax[0, 1].set_title("Min/Max Travel Time")
    ax[0, 1].legend()

    x_vals = np.arange(0, N_ITER)
    nA = {}
    for a in range(N_ACTIONS):
        nA[a] = [M[t]["nA"][a] for t in range(N_ITER)]

    ax[1, 0].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, N_ACTIONS)])

    for a in range(N_ACTIONS):
        ax[1, 0].scatter(x_vals, nA[a], label=a_labels[a], alpha=0.4)
    ax[1, 0].set_ylim((0, N_AGENTS))
    ax[1, 0].set_xlabel('t')
    ax[1, 0].set_ylabel('number of actions')
    ax[1, 0].set_title("Action Profile")
    ax[1, 0].legend()

    Qmean = [M[t]["Qmean"] for t in range(N_ITER)]

    ax[1, 1].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, N_ACTIONS)])

    ax[1, 1].plot(Qmean, label=a_labels)
    ax[1, 1].set_ylim((-2, -1))
    ax[1, 1].set_xlabel('t')
    ax[1, 1].set_ylabel(r'$\hat{Q}(a)$')
    ax[1, 1].set_title(r"$\hat{Q}(a)$ Averaged over Drivers")
    ax[1, 1].legend()

    plt.savefig(NAME + ".png")
    plt.show()


def vecMOrun(N_AGENTS, N_ACTIONS, N_ITER, EPSILON, WEIGHTS, GAMMA, ALPHA, QINIT, PAYOFF_TYPE):

    if type(QINIT) == np.ndarray:
        Q = QINIT.T * np.ones((N_AGENTS, N_ACTIONS))
    elif QINIT == "UNIFORM":
        Q = - np.random.random_sample(size=(N_AGENTS, N_ACTIONS)) - 1

    if WEIGHTS == "RANDOM":
        P = np.random.rand(N_AGENTS)
    elif WEIGHTS == "BETA":
        P = np.random.beta(6, 1, size=(N_AGENTS))
    elif type(WEIGHTS) is float or int:
        P = np.ones(N_AGENTS) * WEIGHTS

    M = {}

    ind = np.arange(N_AGENTS)

    for t in range(N_ITER):

        ## DETERMINE ACTIONS
        rand = np.random.random_sample(size=N_AGENTS)
        # print(rand)
        randA = np.random.randint(N_ACTIONS, size=N_AGENTS)
        # print(randA)
        A = np.where(rand >= EPSILON,
                     np.argmax(Q, axis=1),
                     randA)
        # print(A)
        ## DETERMINE PAYOFFS PER PLAYER
        if N_ACTIONS == 2:
            mean = np.mean(A)
            r_0 = 2 - mean
            r_1 = 1 + mean
            T = [-r_0, -r_1]

            R = -1 * np.where(A == 0, r_0, r_1)

        elif N_ACTIONS == 3:

            n_up = (A == 0).sum()
            n_down = (A == 1).sum()
            n_cross = (A == 2).sum()

            # TIME COSTS
            r_0 = 1 + (n_up + n_cross) / N_AGENTS
            r_1 = 1 + (n_down + n_cross) / N_AGENTS
            r_2 = (n_up + n_cross) / N_AGENTS + (n_down + n_cross) / N_AGENTS
            Tt = [-r_0, -r_1, -r_2]

            dict_t_map = {0: r_0, 1: r_1, 2: r_2}

            Rt = -1 * np.vectorize(dict_t_map.get)(A)

            # VIEW COSTS
            v_0 = (2 * n_up + n_cross) / N_AGENTS
            v_1 = (2 * n_down + n_cross) / N_AGENTS
            v_2 = (n_up + 2 * n_cross + n_down) / N_AGENTS
            Tv = [-v_0, -v_1, -v_2]

            dict_v_map = {0: v_0, 1: v_1, 2: v_2}

            Rv = -1 * np.vectorize(dict_v_map.get)(A)

        if PAYOFF_TYPE == "SOCIAL":
            W = welfare(R)
            R = W * np.ones(N_AGENTS)

        ## UPDATE AGENT Q VALUES
        R = P * Rt + (1 - P) * Rv
        # print(R)
        Q[ind, A] = Q[ind, A] + ALPHA * (R + GAMMA * Q.max(axis=1) - Q[ind, A])
        Qmean = Q.mean(axis=0)
        Wt = welfare(Rt, N_AGENTS)
        Wv = welfare(Rv, N_AGENTS)

        ## UPDATE P VALUES
        # Qmax = Q.max()

        ## SAVE PROGRESS DATA
        # M[t] = {"A": A, "R": R}
        M[t] = {"A": A, "nA": np.bincount(A, minlength=3), "R": Rt, "T": Tt,
                "Qmean": Qmean, "Wt": Wt, "Wv": Wv}
        # print(P)
    return M, Q
