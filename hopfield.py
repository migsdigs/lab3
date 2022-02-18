import itertools
from random import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Hopfield():
    def __init__(self, dimension):
        self.dimension = dimension

        # init symmetric weight matrix - all nodes have correlation with every other node
        self.W = np.zeros((dimension, dimension))

    def train(self, X):
        '''
        Compute weight matrix for hopfield
        '''
        N_dim = np.shape(X)[1]
        W = (1/N_dim)*(X.T @ X)

        # Diagonals should be 0 - node connection with themselves is 0
        np.fill_diagonal(W, 0)
        self.W = W
        return self.W

    def random_weight(self, symmetric=False):
        W_rand = np.random.normal(size=(self.dimension, self.dimension))
        if symmetric:
            self.W = 0.5*(W_rand + W_rand.T)
        else:
            self.W = W_rand

    def plot_weights(self):
        plt.imshow(self.W)
        plt.colorbar()

    def asynchronous_recall(self, x, n_steps):  # One at a time (sequential)
        # Evolution of vector and energy
        xs = []
        energies = []
        energy_old = np.infty
        new_energy = energy(x, self.W)

        count = 0
        while (energy_old > new_energy) and count < n_steps:
            count += 1

            energy_old = new_energy
            xs.append(np.copy(x))
            energies.append(energy_old)

            for ind in np.random.permutation(range(self.dimension)):
                x[:, ind] = np.sign(x @ self.W[:, [ind]])
            # y = x @ self.W
            # x = np.sign(y)

            # new energy
            new_energy = energy(x, self.W)

        return xs, energies, count

    def synchronous_recall(self, x, n_steps):
        # Evolution of vector and energy
        xs = []
        energies = []
        energy_old = np.infty
        new_energy = energy(x, self.W)

        count = 0
        while(energy_old > new_energy) and count < n_steps:
            count += 1

            energy_old = new_energy
            xs.append(x)
            energies.append(energy_old)

            x = np.sign(x @ self.W)
            # y = x @ self.W
            # x = np.sign(y)

            # new energy
            new_energy = energy(x, self.W)

        return xs, energies, count

    def find_attractors(self, n_steps, n_attr=0):
        if n_attr:
            possible_inputs = np.array(list(itertools.islice(
                itertools.product([-1, 1], repeat=self.dimension), n_attr)))

        else:
            possible_inputs = np.array(
                list(itertools.product([-1, 1], repeat=self.dimension)))
        n_combs = np.shape(possible_inputs)[0]
        print(n_combs)

        attractors = []
        attractor_energies = []
        for i in range(n_combs):
            x = possible_inputs[[i], :]

            xs, energies, count = self.asynchronous_recall(x, n_steps)
            attractor = list(xs[-1].reshape(-1))
            energy_at_attractor = energies[-1]

            if attractor not in attractors:
                attractors.append(attractor)
                attractor_energies.append(energy_at_attractor)

        attractors = np.array(attractors)

        return attractors, attractor_energies


def energy(x, W):
    return float(-0.5 * x @ W @ x.T)


def plot_energies(energy_list, name=''):
    n_patterns = len(energy_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_patterns):
        ax.plot(energy_list[i], label='x{}{}'.format(i+1, name))
    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy")
    ax.legend()


def plot_energies_noisy_img(energy_list, name=''):
    n_patterns = len(energy_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_patterns):
        ax.plot(energy_list[i], label='x-{}%-{}'.format(i*10, name))
    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy")
    ax.legend()


def generate_noisy_image(x):
    dimension = np.shape(x)[0]
    noise_percentage = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # noise_percentage = [0.1]
    n_noise_bits = []
    noisy_images = []
    x_noise = x.copy()
    for i in range(len(noise_percentage)):
        n_noise_bits.append(int(noise_percentage[i]*dimension))

    for j in range(len(n_noise_bits)):

        n_bits_to_invert = n_noise_bits[j]
        random_ind = np.random.permutation(dimension)
        random_ind = list(random_ind[:n_bits_to_invert])

        if len(random_ind) != 0:    # To account for 0 noise case
            x_noise = x.copy()
            # invert bits
            for k in random_ind:
                x_noise[k] = -1 * x[k]
        noisy_images.append(x_noise)

    noisy_images = np.array(noisy_images)
    return noisy_images


def distort_patterns(X, n_bits):
    dimension = np.shape(X)[1]
    n_patterns = np.shape(X)[0]

    if n_bits > dimension:
        n_bits = dimension

    Xd = np.copy(X)

    for i in range(n_patterns):
        ind = np.random.permutation(dimension)
        # print(ind)
        for bit in range(n_bits):
            Xd[i][ind[bit]] = -1*Xd[i][ind[bit]]

    return Xd


def random_patterns(n_patterns, dim):
    '''
    Create an array of random binary 
    '''
    X_rand = np.random.randint(0, 2, size=(n_patterns, dim))*2 - 1
    return X_rand


def show_patterns(X, name=''):
    '''
    Note: X is passed in as a set of column vectors
    '''
    n_patterns = np.shape(X)[1]
    fig, ax = plt.subplots(1, n_patterns, figsize=(18, 6))
    for i in range(n_patterns):
        ax[i].imshow(X[:, [i]], cmap='binary')
        ax[i].set_title('x{}{}'.format(i+1, name))


def compare_patterns(X, Xd, name=''):
    n_patterns = np.shape(X)[1]
    fig, ax = plt.subplots(1, 2*n_patterns, figsize=(18, 6))
    for i in range(int(n_patterns)):
        ax[2*i].imshow(X[:, [i]], cmap='binary')
        ax[2*i].set_title('x{}{}'.format(i+1, name))
        ax[2*i+1].imshow(Xd[:, [i]], cmap='binary')
        ax[2*i+1].set_title('x{}{}'.format(i+1, name))


def show_img_pattern(X, name=''):
    n_patterns = np.shape(X)[0]
    dim = np.shape(X)[1]
    image_size = int(np.sqrt(dim))
    fix, ax = plt.subplots(1, n_patterns, figsize=(18, 6))
    for i in range(n_patterns):
        ax[i].imshow(X[[i], :].reshape(
            (image_size, image_size)).T, cmap='binary')
        ax[i].set_title('x{}{}'.format(i+1, name))


def compare_img_patterns(X, Xd, recovered=False, name=''):
    n_patterns = np.shape(X)[0]
    dim = np.shape(X)[1]
    image_size = int(np.sqrt(dim))
    fig, ax = plt.subplots(n_patterns, 2, figsize=(8, 14))
    for i in range(int(n_patterns)):
        ax[i, 0].imshow(Xd[[i], :].reshape(
            (image_size, image_size)).T, cmap='binary')
        ax[i, 0].set_title('x{}{}'.format(i+1, name))
        ax[i, 1].imshow(X[[i], :].reshape(
            (image_size, image_size)).T, cmap='binary')
        ax[i, 1].set_title('x{}d'.format(i+1) + ('_rec' if recovered else ''))


def compare_noisy_img_patterns(X_rec, Xd, recovered=False, name=''):
    n_patterns = np.shape(X_rec)[0]
    dim = np.shape(X_rec)[1]
    image_size = int(np.sqrt(dim))
    fig, ax = plt.subplots(n_patterns, 2, figsize=(8, 6*n_patterns))
    for i in range(int(n_patterns)):
        ax[i, 0].imshow(Xd[[i], :].reshape(
            (image_size, image_size)).T, cmap='binary')
        ax[i, 0].set_title('x_{}%_{}'.format(i*10, name))
        ax[i, 1].imshow(X_rec[[i], :].reshape(
            (image_size, image_size)).T, cmap='binary')
        ax[i, 1].set_title('x_{}%'.format(i*10) +
                           ('_rec' if recovered else ''))
