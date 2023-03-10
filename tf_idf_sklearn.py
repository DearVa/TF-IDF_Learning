from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)  # 直接使用sklearn获取tf_idf

q = "I get a coffee cup"
q_tf_idf = vectorizer.transform([q])
res = cosine_similarity(tf_idf, q_tf_idf)
res = res.ravel().argsort()[-3:]

print("top 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))

i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
show_tfidf(tf_idf.todense(), [i2v[i] for i in range(len(i2v))], "tf_idf")
