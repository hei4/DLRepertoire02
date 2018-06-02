# DLRepertoire02
more deep learning repertoire #2

# Data
def make_sin_data(data_per_cycle=200, n_cycle=5):
    np.random.seed(0)

    n_data = n_cycle * data_per_cycle
    theta = np.linspace(0., n_cycle * (2. * np.pi), num=n_data)

    X = np.sin(theta) + 0.1 * (2. * np.random.rand(n_data) - 1.)
    X /= np.std(X)

    return X.astype(np.float32)

