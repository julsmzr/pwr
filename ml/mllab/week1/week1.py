import numpy as np
import matplotlib.pyplot as plt


def task1():
    arr = np.ones(shape=(3,5,7), dtype=np.int8)
    print(arr)
    print('-' * 50)
    print("Task 1 Output\n")
    print(np.sum(arr, axis=0), "\n")
    print(np.sum(arr, axis=1), "\n")
    print(np.sum(arr, axis=2), "\n")
    print('-' * 50)

def task2():
    a1 = np.random.normal(loc=0.0, scale=1.0, size=(130, 2))
    a2 = np.random.normal(loc=1.0, scale=3.0, size=(30,2))

    print("Task 2 Output\n")
    print(a1[-1])
    print()
    print(a2[:, 0], "\n")
    print('-' * 50)
    return a1, a2

def task3(a1, a2):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax.scatter(x=a1[:, 0], y=a1[:, 1], c='r')
    ax.scatter(x=a2[:, 0], y=a2[:, 1], c='b')

    fig.tight_layout()
    plt.grid()
    plt.savefig("plot.png")

    print("Task 3 Output\n")
    print("Plot successfully saved to 'plot.png'")
    print('-' * 50, "\n")


if __name__ == "__main__":
    print("Running tasks...\n")

    task1()
    a1, a2 = task2()
    task3(a1, a2)

    print("All tasks completed.")
