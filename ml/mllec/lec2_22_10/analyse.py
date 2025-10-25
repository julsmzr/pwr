# How to work with high dimensional data and analyse it after an experiment

import numpy as np

# How to load data from saved numpy matrix files.
def load_data():
    scores = np.load("data/experiment_results.npy")
    return scores

def generate_and_data():
    # DATASETS x CLF x FOLDS x PROCESSING x BUDGET <-- this is best practise!
    scores = np.random.randint(-9999, 9999, (20, 8, 10, 15, 40))
    print(scores.shape)

    # DATASETS x CLF x PROCESSING x BUDGET
    scores = np.mean(scores, axis=2) # folds disappear
    print(scores.shape) 

    # DATASETS x CLF x BUDGET
    scores = scores[:, :, 0] # only take first processing type
    print(scores.shape)

    return scores

if __name__ == "__main__":

    # scores = load_data()
    scores = generate_and_data()

