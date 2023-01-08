from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset, load_from_disk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

countries = set(map(lambda x: x.lower(), {
    "Albania",
    "Algeria",
    "Angola",
    "Anguilla",
    "Argentina",
    "Armenia",
    "Aruba",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bermuda",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei Darussalam",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Cape Verde",
    "Cayman Islands",
    "Central African Rep",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Congo",
    "Cook Islands",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cyprus",
    "Czech (Rep)",
    "Luxembourg",
    "Macao",
    "Macedonia",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Mauritius",
    "Mauritania",
    "Mexico",
    "Moldova",
    "Mongolia",
    "Morocco",
    "Myanmar",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Caledonia",
    "New Zealand",
    "Niger",
    "Nigeria",
    "Norway",
    "Oman",
    "Pakistan",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russian Federation",
    "Rwanda",
    "Saint Christopher and Nevis",
    "Dem Rep of Congo",
    "Denmark",
    "Dominican Rep.",
    "Dominica",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Eritrea",
    "Estonia",
    "Ethiopia",
    "Fiji",
    "Finland",
    "French Polynesia",
    "France",
    "Gabon",
    "Georgia",
    "Germany",
    "Ghana",
    "Gibraltar",
    "United Kingdom of Great Britain and Northern Ireland",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kenya",
    "Korea",
    "Kuwait",
    "Lao People's Dem Rep",
    "Latvia",
    "Lesotho",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Swaziland",
    "Sweden",
    "Switzerland",
    "Syrian Arab Rep",
    "Tanzania",
    "Thailand",
    "Togo",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United States of America",
    "Uruguay",
    "Venezuela",
    "Viet Nam",
    "Western Samoa",
    "Yemen",
    "Yugoslavia",
    "Zambia",
    "Zimbabwe"
}))
names = set(map(lambda x: x.lower(), {
    "Oliver",
    "Jack",
    "Harry",
    "Jacob",
    "Charlie",
    "Thomas",
    "George",
    "Oscar",
    "James",
    "William",
    "Amelia",
    "Olivia",
    "Isla",
    "Emily",
    "Poppy",
    "Ava",
    "Isabella",
    "Jessica",
    "Lily",
    "Sophie",
    "Jake",
    "Connor",
    "Callum",
    "Jacob",
    "Kyle",
    "Joe",
    "Reece",
    "Rhys",
    "Charlie",
    "Damian",
    "Margaret",
    "Samantha",
    "Bethany",
    "Elizabeth",
    "Joanne",
    "Megan",
    "Victoria",
    "Lauren",
    "Michelle",
    "Tracy",
    "Noah",
    "Liam",
    "Mason",
    "Jacob",
    "William",
    "Ethan",
    "Michael",
    "Alexander",
    "James",
    "Daniel",
    "Emma",
    "Olivia",
    "Sophia",
    "Isabella",
    "Ava",
    "Mia",
    "Emily",
    "Abigail",
    "Madison",
    "Charlotte",
    "James",
    "John",
    "Robert",
    "Michael",
    "William",
    "David",
    "Richard",
    "Joseph",
    "Charles",
    "Thomas",
    "Mary",
    "Patricia",
    "Jennifer",
    "Elizabeth",
    "Linda",
    "Barbara",
    "Susan",
    "Margaret",
    "Jessica",
    "Sarah"
}))

dataset_path = '.\\wiki_country_name.en'


def prepare():
    """提取出含有国家和人名的词条"""
    raw_wiki = load_dataset("wikipedia", "20220301.en")['train']
    all_words = countries.union(names)
    my_dataset = raw_wiki.filter(lambda example: any(x in example['title'].lower() for x in all_words))

    print(my_dataset.shape)
    my_dataset.save_to_disk(dataset_path)


def visualize():
    wiki = load_from_disk(dataset_path).shuffle()  # 读取预处理数据并打乱

    country_wikis = {}
    name_wikis = {}
    idx = 0
    max_count = 20
    # 分别找出20个代表国家和人物的句子
    while idx < wiki.num_rows and len(country_wikis) < max_count or len(name_wikis) < max_count:
        title = str(wiki[idx]['title']).lower()
        if len(country_wikis) < max_count and title in countries:  # 是国家吗？这里使用严格的全词匹配
            country_wikis[title] = wiki[idx]['text']
        elif len(name_wikis) < max_count and title.find('(') != -1 and 0 < title.count(' ') < 5 \
                and any(x in title for x in names):
            # 是人物吗？采用的方法是词的个数在2~6个，并且带有括号，形如'Richard h. Robinson (california politician)'
            name_wikis[title] = wiki[idx]['text']

        idx += 1

    #  使用两个TfidfVectorizer求出各自的TF-IDF
    country_vectorizer = TfidfVectorizer()
    country_tf_idf = country_vectorizer.fit_transform(country_wikis.values())
    name_vectorizer = TfidfVectorizer()
    name_tf_idf = name_vectorizer.fit_transform(name_wikis.values())
    # 打印出各自的矩阵大小。这里大概率两者是不一样的，所以我采用了如下求交集的方法
    print(country_tf_idf.shape)
    print(name_tf_idf.shape)

    # 求出两者共同拥有的词语
    same_words = set(country_vectorizer.vocabulary_.keys()).intersection(name_vectorizer.vocabulary_.keys())
    print('same words: {}'.format(same_words))

    # 将原有的矩阵提取出相同词语的列，这样就变成了两个大小完全一样的矩阵
    country_indices = [country_vectorizer.vocabulary_[x] for x in same_words]
    country_indices.sort()
    country_tf_idf_vec_array = np.asarray(country_tf_idf[:, country_indices].todense().tolist())  # 转向量数组以便可视化
    name_indices = [name_vectorizer.vocabulary_[x] for x in same_words]
    name_indices.sort()
    name_tf_idf_vec_array = np.asarray(name_tf_idf[:, name_indices].todense().tolist())

    ts = TSNE(perplexity=10, n_components=2, init='pca', random_state=0)
    # 合并后一次性转换到二维空间
    tsne_data = ts.fit_transform(np.concatenate((country_tf_idf_vec_array, name_tf_idf_vec_array), axis=0))

    # 开始进行可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_data[:20, 0], tsne_data[:20, 1], label='country')
    plt.scatter(tsne_data[20:, 0], tsne_data[20:, 1], label='name')
    plt.show()


visualize()
