import matplotlib.pyplot as plt
import numpy as np
import os


def show_tfidf(tfidf, vocab, filename):
    """
    在matplotlib显示tf_idf的图像
    """
    # [n_doc, n_vocab]
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocab, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0] + 1), fontsize=6)
    plt.tight_layout()
    # creating the output folder
    output_folder = './visual/results/'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, '%s.png') % filename, format="png", dpi=512)
    plt.show()