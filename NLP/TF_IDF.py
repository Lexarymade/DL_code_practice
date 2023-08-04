from sklearn.feature_extraction.text import TfidfVectorizer

#实例化
tfidf_vec = TfidfVectorizer()

documents = [
    'this is the bayes document',
    'The first album of Lexarymade is made in luxury',
    'and the third one',
    'Roger Federer is the goat'
]

model = tfidf_vec.fit_transform(documents)
print('输出每个单词对应的 id 值:', tfidf_vec.vocabulary_)
print('不重复的词:', tfidf_vec.get_feature_names_out())
print('输出每个单词在每个文档中的 TF-IDF 值，向量里的顺序是按照词语的 id 顺序来的:', '\n', model.toarray())



