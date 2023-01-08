import numpy as np
from collections import Counter
import itertools
from utils import show_tfidf


docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",", "").split(" ") for d in docs]
vocab = set(itertools.chain(*docs_words))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x


tf_methods = {
    "log": lambda x: np.log(1 + x),
    "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
    "boolean": lambda x: np.minimum(x, 1),
    "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
}

idf_methods = {
    "log": lambda x: 1 + np.log(len(docs) / (x + 1)),
    "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x + 1))),
    "len_norm": lambda x: x / (np.sum(np.square(x)) + 1),
}


def get_tf(method="log"):
    tf_mat = np.zeros((len(vocab), len(docs)), dtype=np.float64)
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        most_common_word_count = counter.most_common(1)[0][1]
        for v in counter.keys():
            tf_mat[v2i[v], i] = counter[v] / most_common_word_count

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(tf_mat)


def get_idf(method="log"):
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count

    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)


def cosine_similarity(query, _tf_idf):
    unit_q = query / np.sqrt(np.sum(np.square(query), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


tf = get_tf()           # [n_vocab, n_doc]
idf = get_idf()         # [n_vocab, 1]
tf_idf = tf * idf       # [n_vocab, n_doc]
print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])


def docs_score(query: str, len_norm=False):
    q_words = query.replace(",", "").split(" ")

    # 可能存在原始语料库中没有的词语
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1  # 求出未知词语的数量
    if unknown_v > 0:  # 将原始矩阵升维
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=float)     # 维度为[n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf            # query的向量，作为矩阵处理，维度为[n_vocab, 1]

    q_scores = cosine_similarity(q_vec, _tf_idf)  # 求cosine相似度
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def get_keywords(n=2):
    for c in range(3):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))


get_keywords()
q = "I get a coffee cup"
scores = docs_score(q)
d_ids = scores.argsort()[::-1]
print("\ntop docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))

show_tfidf(tf_idf.T, [i2v[i] for i in range(tf_idf.shape[0])], "tfidf_matrix")
