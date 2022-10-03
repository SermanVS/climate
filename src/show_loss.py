import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output

def show_loss(running_loss, epoch_loss, clear_after=True):

    plt.figure(figsize=[18, 6])
    plt.subplot(1, 3, 1)
    plt.scatter(np.arange(start=1, stop=len(running_loss) + 1, step=1), running_loss, s=6, alpha = 0.6)
    plt.xlim((1, len(running_loss) + 1))
    plt.ylim((0, max(running_loss)))
    plt.title("General loss change")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(start=len(running_loss) - 99, stop=len(running_loss) + 1, step=1), running_loss[-100:])
    plt.xlim((len(running_loss) - 99, len(running_loss) + 1))
    plt.ylim((0, max(running_loss)))
    plt.title("Loss change in window 100")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(start=1, stop=len(epoch_loss) + 1, step=1), epoch_loss, 'o-')
    plt.xlim((1, len(epoch_loss) + 2))
    plt.xticks(np.arange(start=1, stop=len(epoch_loss) + 3, step=1))
    plt.ylim((0, max(running_loss)))
    plt.title("Epoch mean loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()

    if clear_after:
        clear_output(True)

    plt.show()
