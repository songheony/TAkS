import numpy as np
from numpy.testing import assert_array_almost_equal


# flipping code from https://github.com/hongxin001/JoCoR
def multiclass_noisify(y, P, random_state):
    """Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state, nb_classes=10):
    """mistakes:
    flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1.0 - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_multiclass_symmetric(y_train, noise, random_state, nb_classes=10):
    """mistakes:
    flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1.0 - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1.0 - n
        P[nb_classes - 1, nb_classes - 1] = 1.0 - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_mnist_asymmetric(y_train, noise, random_state):
    """mistakes:
    1 <- 7
    2 -> 7
    3 -> 8
    5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1.0 - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1.0 - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1.0 - n, n
        P[6, 6], P[6, 5] = 1.0 - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_cifar10_asymmetric(y_train, noise, random_state):
    """mistakes:
    automobile <- truck
    bird -> airplane
    cat <-> dog
    deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1.0 - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1.0 - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1.0 - n, n
        P[5, 5], P[5, 3] = 1.0 - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1.0 - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def build_for_cifar100(size, noise):
    """The noise matrix flips to the "next" class with probability 'noise'."""

    assert (noise >= 0.0) and (noise <= 1.0)

    P = (1.0 - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def noisify_cifar100_asymmetric(y_train, noise, random_state):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'"""
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print("Actual noise %.2f" % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_rand(y_train, noise, random_state, num_classes):
    """code from https://github.com/scifancier/Class2Simi
    """
    t = np.random.RandomState(random_state).rand(num_classes, num_classes)
    i = np.eye(num_classes)
    if noise == 0.1:
        t = t + 3.0 * num_classes * i
    if noise == 0.2:
        t = t + 1.7 * num_classes * i
    if noise == 0.3:
        t = t + 1.2 * num_classes * i
    if noise == 0.4:
        t = t + 0.6 * num_classes * i
    if noise == 0.5:
        t = t + 0.4 * num_classes * i
    if noise == 0.6:
        t = t + 0.24 * num_classes * i
    for a in range(num_classes):
        t[a] = t[a] / t[a].sum()

    P = np.asarray(t)

    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify(dataset_name, nb_classes, train_labels, noise_type, noise_rate, random_state):
    if noise_type == "pairflip":
        train_noisy_labels, P = noisify_pairflip(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes
        )
    elif noise_type == "symmetric":
        train_noisy_labels, P = noisify_multiclass_symmetric(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes
        )
    elif noise_type == "asymmetric":
        if dataset_name == "mnist":
            train_noisy_labels, P = noisify_rand(train_labels, noise_rate, random_state=random_state, num_classes=nb_classes)
        elif dataset_name == "cifar10":
            train_noisy_labels, P = noisify_rand(train_labels, noise_rate, random_state=random_state, num_classes=nb_classes)
        elif dataset_name == "cifar100":
            train_noisy_labels, P = noisify_rand(
                train_labels, noise_rate, random_state=random_state, num_classes=nb_classes
            )
    elif noise_type == "hard_asymmetric":
        if dataset_name == "mnist":
            train_noisy_labels, P = noisify_mnist_asymmetric(train_labels, noise_rate, random_state=random_state)
        elif dataset_name == "cifar10":
            train_noisy_labels, P = noisify_cifar10_asymmetric(train_labels, noise_rate, random_state=random_state)
        elif dataset_name == "cifar100":
            train_noisy_labels, P = noisify_cifar100_asymmetric(
                train_labels, noise_rate, random_state=random_state
            )
    return train_noisy_labels, P
