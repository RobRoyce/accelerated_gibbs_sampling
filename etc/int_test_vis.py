import numpy as np
import matplotlib.pyplot as plt

read_array = lambda type: np.array(list(map(type, input().split())))

def gaussian_pdf(mu, sigma2):
   return lambda x: np.exp(-(x - mu)**2/sigma2)/np.sqrt(2*np.pi * sigma2)

if __name__ == '__main__':
    k = int(input())
    comp_params = [read_array(float) for _ in range(k)]
    data = read_array(float)
    zs = read_array(int)

    plt.figure()
    cmap = plt.get_cmap('Paired')
    for j in range(k):
        w, mu, sigma2 = comp_params[j]
        x = np.linspace(mu - 3*sigma2**.5, mu + 3*sigma2**.5, 200)
        comp_pts = data[zs == j]
        plt.hist(comp_pts, density=True, bins=1, color=cmap(2*j), alpha=0.5)
        plt.plot(x, gaussian_pdf(mu, sigma2)(x), color=cmap(2*j+1))
        plt.plot(comp_pts, 0.1*np.ones_like(comp_pts), '.', color=cmap(2*j+1))
        plt.ylim(0,0.4)
    plt.savefig('plot.png')




