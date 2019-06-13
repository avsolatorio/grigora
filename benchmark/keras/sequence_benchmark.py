import numpy as np
import pandas as pd

from grigora.keras.preprocessing.sequence import skipgrams as grigora_skipgrams
from keras.preprocessing.sequence import skipgrams as keras_skipgrams


def generate_data(
    vocabulary_size=10000,
    seqlen=500,
    window_size=5,
    negative_samples=5,
    seed=1029,
    shuffle=True,
    ):
    np.random.seed(seed)
    sequence = np.random.randint(1, vocabulary_size, size=seqlen)

    return dict(
        sequence=list(sequence),
        vocabulary_size=vocabulary_size,
        window_size=window_size,
        negative_samples=negative_samples,
        seed=seed,
        shuffle=shuffle
    )


def benchmark_grigora_skipgrams(data):
    grigora_skipgrams(**data)


def benchmark_keras_skipgrams(data):
    keras_skipgrams(**data)


def compute_speedup(iter_count=1):
    keras_times = []
    grigora_times = []

    for i in range(iter_count):
        data = generate_data(
            vocabulary_size=10000,
            seqlen=1000, seed=i
        )

        start = time.time()
        benchmark_keras_skipgrams(data)
        keras_times.append(time.time() - start)

        start = time.time()
        benchmark_grigora_skipgrams(data)
        grigora_times.append(time.time() - start)

    keras_times = pd.Series(keras_times)
    grigora_times = pd.Series(grigora_times)

    print(f"Keras performance summary:\n{keras_times.describe()}\n")
    print(f"Grigora performance summary:\n{grigora_times.describe()}\n")

    return dict(
        keras_times=keras_times,
        grigora_times=grigora_times
    )


if __name__ == '__main__':
    import timeit
    import time

    iter_count = 100
    perf = compute_speedup(iter_count)

    print(f"Speedup: {perf['keras_times'].mean() / perf['grigora_times'].mean():.2f}x")
