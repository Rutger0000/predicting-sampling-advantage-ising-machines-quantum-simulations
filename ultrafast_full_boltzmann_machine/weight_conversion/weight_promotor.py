import numpy as np
import matplotlib.pyplot as plt
import os

def promote_weights(nspins, alpha, directory):
    # check whether data/models/{directory}/W_ising_{nspins}_{alpha}_ti_W.csv exists
    # if not, then we need to convert the weights from the RBM to the Ising model

    output_path_W = f'data/models/{directory}/W_ising_{nspins}_{alpha}_ti_W.csv'
    output_path_b = f'data/models/{directory}/W_ising_{nspins}_{alpha}_ti_b.csv'
    output_path_J = f'data/models/{directory}/Ising_{nspins}_{alpha}_ti_J.csv'
    output_path_h = f'data/models/{directory}/Ising_{nspins}_{alpha}_ti_h.csv'

    if os.path.exists(output_path_W) and os.path.exists(output_path_b) and os.path.exists(output_path_J) and os.path.exists(output_path_h):
        print(f"Ignore file {nspins}, {alpha}, {directory} exists")
        return

    W_old = np.loadtxt(f'data/models/{directory}/W_RBM_{nspins}_{alpha}_ti_W.csv', delimiter=',')
    b_old = np.loadtxt(f'data/models/{directory}/W_RBM_{nspins}_{alpha}_ti_b.csv', delimiter=',')


    M = W_old.shape[1]
    N = W_old.shape[0]

    print(f"Note that M = {M}, N = {N} and alpha = {alpha} where M = N x alpha = {nspins} x {alpha} and bias size = M = {b_old.shape[0]}")

    W_new = np.block([
        [np.zeros((M, M)), W_old.T/2],
        [W_old/2, np.zeros((N, N))]
    ]
    )

    J_new = np.block([
        [np.zeros((M, M)), W_old.T],
        [W_old, np.zeros((N, N))]
    ]
    )

    h_new = np.concatenate((b_old, np.zeros(N)))
    b_new = np.concatenate((b_old, np.zeros(nspins)))

    # Plotting the matrix
    W_new_masked = np.ma.masked_where(W_new == 0, W_new)

    plt.matshow(W_new_masked)

    plt.xticks([0, M-1, M+N-1], ['$h_1$', '$h_M$', '$S_N$'])
    plt.yticks([0, M, M+N-1], ['$h_1$', '$S_1$', '$S_N$'])
    plt.title(f'W matrix for RBM with {nspins} spins and alpha = {alpha}, with $M$ = {M}, $N$ = {N}, $M+N$ = {M+N}')

    # create directory if it does not exist
    if not os.path.exists(f'data/models/{directory}/figures/'):
        os.makedirs(f'data/models/{directory}/figures/')

    plt.savefig(f'data/models/{directory}/figures/W_RBM_{nspins}_{alpha}_ti_W.pdf', bbox_inches='tight')

    # Saving the exchange matrices in different formats
    print("shape of W_new = ", W_new.shape)
    print("shape of b_new = ", b_new.shape)

    # W_ising works with full summation so i=1 to N and j=1 to N
    np.savetxt(output_path_W, W_new, delimiter=',')
    np.savetxt(output_path_b, b_new, delimiter=',')

    # J_ising works with half summation so i=1 to N and j=i+1 to N
    np.savetxt(output_path_J, J_new, delimiter=',')
    np.savetxt(output_path_h, h_new, delimiter=',')


