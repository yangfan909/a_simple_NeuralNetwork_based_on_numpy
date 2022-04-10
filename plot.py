import os.path

import numpy as np

from NetWork import *
import matplotlib.pyplot as plt
import seaborn as sns

a = Model()
a.load_model("./test/final/model.h5")
path = "./test/final/pict/"
if not os.path.exists(path):
    os.mkdir(path)
layer = a.layer
for item in layer:
    if item.layer == 'Dense':
        pt = item.weights
        co, ci = pt.shape
        f, ax1 = plt.subplots(figsize=(ci, co))

        # cmap用matplotlib colormap
        try:
            sns.heatmap(pt, linewidths=0.05, ax=ax1, cmap='rainbow')
        except ValueError:
            print("图片太大无法显示\n")
            continue
        # rainbow为 matplotlib 的colormap名称
        ax1.set_title('matplotlib colormap',fontdict={'weight': 'normal', 'size': 5})
        ax1.set_xlabel('region')
        ax1.set_ylabel('kind')
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=10)
        cbar = ax1.collections[0].colorbar
        cbar.set_label(r'$NMI$', fontdict={'size': 10})
        plt.savefig(path + item.name + "_weights.jpeg")
        plt.show()
        print(item.name)
    if item.layer == 'Convolution2D':
        pt = item.weights
        co, ci, k_h, k_w = item.weights.shape

        for i in range(co):
            f, axs = plt.subplots(dpi=600, figsize=(k_h*ci, k_w), ncols=ci)
            if ci == 1:
                temp = np.array(pt[i, 0, :, :].copy())
                sns.heatmap(temp, linewidth=0.05, ax=axs, cmap='rainbow')
                axs.set_title('weights[%d,%d]' % (i, 0), fontdict={'weight': 'normal', 'size': 4})
                cax = plt.gcf().axes[-1]
                cax.tick_params(labelsize=4)
            else:
                axs = axs.flatten()
                for j in range(ci):

                    temp = np.array(pt[i, j, :, :].copy())
                    sns.heatmap(temp, linewidth=0.05, ax=axs[j],cmap='rainbow')
                    axs[j].set_title('weights[%d,%d]' % (i, j), fontdict={'weight': 'normal', 'size': 4})
                    cax = plt.gcf().axes[-1]
                    cax.tick_params(labelsize=4)
                    cbar = axs[j].collections[0].colorbar
                    cbar.set_label(r'$NMI$', fontdict={'size': 4})
            plt.savefig(path + item.name + "_weights[%d].jpeg"%i, dpi=600)

            plt.show()
            print(item.name)
