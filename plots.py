
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
    
class plot_spectrogram_to_numpy():
    def __call__(self, spectrogram):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()
        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data