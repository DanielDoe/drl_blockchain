import numpy as np
import matplotlib.pyplot as plt


def plot_sat():
    lamda_k = [100, 100, 100, 100]  # packets per second
    packet_size = [400, 4000, 500, 120]  # bit
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    beta = 1
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        arrive_rate = 0.75 * lamda_k[si] * packet_size[si] / 1e3
        x = np.arange(0, 2, 0.1)
        if si != 3:
            ue_rate = arrive_rate * x
            y1 = 1 / (1 + np.exp(1 * (arrive_rate - ue_rate)))
            y2 = 1 / (1 + np.exp(1e-1 * (arrive_rate - ue_rate)))
            y3 = 1 / (1 + np.exp(1e-2 * (arrive_rate - ue_rate)))
            y4 = 1 / (1 + np.exp(1e-3 * (arrive_rate - ue_rate)))
        else:
            t_u = 10 * x
            y1 = 1 / (1 + np.exp(1 * (t_u * 0.8 - 10)))
            y2 = 1 / (1 + np.exp(1e-1 * (t_u * 0.8 - 10)))
            y3 = 1 / (1 + np.exp(1e-2 * (t_u * 0.8 - 10)))
            y4 = 1 / (1 + np.exp(1e-3 * (t_u * 0.8 - 10)))
        plt.plot(x, y1, '-+', label='beta=1')
        plt.plot(x, y2, '-+', label='beta=1e-1')
        plt.plot(x, y3, '-+', label='beta=1e-2')
        plt.plot(x, y4, '-+', label='beta=1e-3')
        plt.legend()
        if si != 3:
            plt.xlabel('ue_rate')
        else:
            plt.xlabel('t_u')
        plt.ylabel('sat')
    plt.show()


if __name__ == '__main__':
    plot_sat()
