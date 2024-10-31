import re
import nltk
import string
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


with open("alice.txt", "r", encoding='utf-8') as file:
    alice_text = file.read()


def pipeline(*funcs):
    def inner(data):
        for func in funcs:
            data = func(data)
        return data

    return inner


def convert_lower_case(data):
    return data.lower()


def remove_numbers(data):
    return re.sub(r'\d+', '', data)


def remove_punctuation(data):
    return data.translate(str.maketrans("", "", string.punctuation))


def remove_apostrophe(data):
    return data.replace("’", " ")


def remove_stop_words(data):
    stop_words = set(stopwords.words("english"))
    words_tokenized = nltk.word_tokenize(data)
    return [word for word in words_tokenized if word not in stop_words and word.isalpha()]


def lemmatize(data):
    wnl = WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(word) for word in data])


def divide_in_chapters(data):
    return ["chapter " + chapter.strip() for chapter in data.split("chapter")[13:]]


alice_text = pipeline(convert_lower_case, remove_numbers, remove_punctuation, remove_apostrophe, remove_stop_words,
                      lemmatize)(alice_text)

chapters = divide_in_chapters(alice_text)

print('-------------------- TFIDF --------------------')
tfidf = TfidfVectorizer(stop_words=stopwords.words("english"))

top_words_with_frequency_by_chapter = []

for chapter in chapters:
    tfidf_matrix = tfidf.fit_transform([chapter])
    tfidf_scores = dict(zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray().flatten()))

    sorted_words = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    top_20_words_with_freq = [(word, round(score, 3)) for word, score in sorted_words[:20]]
    top_words_with_frequency_by_chapter.append(top_20_words_with_freq)

for i, words in enumerate(top_words_with_frequency_by_chapter):
    print(f"Top 20 words in Chapter {i + 1} with TF-IDF:")
    for word, freq in words:
        print(f"{word}: {freq}", sep=' ', end='; ', flush=True)
    print('\n')

print('-------------------- LDA --------------------')
tokenized_chapters = [[word for word in chapter.split()] for chapter in chapters]

dictionary = corpora.Dictionary(tokenized_chapters)
corpus = [dictionary.doc2bow(text) for text in tokenized_chapters]

lda_model = LdaModel(corpus, num_topics=20, id2word=dictionary, passes=10)

# for i, topic in sorted(lda_model.show_topics(num_words=20)):
#    print('Topic: {} \nWords: {}'.format(i+1, topic))

for i, chapter in enumerate(corpus):
    topics = lda_model.get_document_topics(chapter)
    for idx, topic in topics:
        print('Topic: {} Chapter: {}'.format(idx, i + 1))
        print(
            f"Words: {[(dictionary[word_id], round(score, 3)) for word_id, score in lda_model.get_topic_terms(topics[0][0], 20)]}\n")

# Here are lists of Top 20 words in Chapter 1

# First try TFIDF
# ['alice', 'little', 'like', 'think', 'way', 'll', 'said', 'thought',
# 'time', 'door', 'eat', 'went', 'going', 'key', 'rabbit', 'say', 'wonder', 'dinah', 'shall',
# 'suddenly']

# Second try TFIDF
# alice: 44.3%; little: 23.7%; like: 17.4%; think: 17.4%; way: 17.4%; see: 15.8%; one: 14.2%;
# could: 12.6%; said: 12.6%; thought: 12.6%; time: 12.6%; door: 11.1%; eat: 11.1%; found: 11.1%;
# get: 11.1%; nothing: 11.1%; well: 11.1%; went: 11.1%; would: 11.1%; either: 9.5%;

# Third try TFIDF
# alice: 43.9%; little: 23.5%; like: 17.3%; way: 17.3%; see: 15.7%; door: 14.1%; one: 14.1%;
# think: 14.1%; could: 12.5%; said: 12.5%; thought: 12.5%; time: 12.5%; eat: 11.0%; found: 11.0%;
# get: 11.0%; nothing: 11.0%; thing: 11.0%; well: 11.0%; went: 11.0%; would: 11.0%;

# First try LDA
# [('said', 0.032), ('alice', 0.028), ('little', 0.009), ('one', 0.007),
# ('like', 0.006), ('would', 0.006), ('went', 0.006), ('know', 0.006), ('could', 0.006), ('thought', 0.005)]

# Second try LDA
# [('alice', 0.032), ('caterpillar', 0.007), ('could', 0.005), ('dormouse', 0.008), ('hare', 0.006),
# ('hatter', 0.009), ('know', 0.006), ('like', 0.006), ('little', 0.01), ('march', 0.006), ('one', 0.008),
# ('said', 0.034), ('say', 0.005), ('see', 0.006), ('thing', 0.006), ('think', 0.006), ('thought', 0.005),
# ('time', 0.009), ('well', 0.008), ('went', 0.005)]

# Third try LDA
# [('said', 0.035), ('alice', 0.034), ('mouse', 0.012), ('hatter', 0.011), ('little', 0.01), ('know', 0.01),
# ('thing', 0.009), ('dormouse', 0.009), ('time', 0.009), ('one', 0.009), ('march', 0.007), ('hare', 0.007),
# ('went', 0.007), ('go', 0.007), ('say', 0.006), ('could', 0.006), ('like', 0.006), ('thought', 0.006),
# ('dear', 0.005), ('must', 0.005)]

# Forth try LDA
# [('alice', 0.03), ('said', 0.028), ('little', 0.013), ('mouse', 0.011), ('one', 0.009), ('like', 0.007),
# ('know', 0.007), ('thought', 0.007), ('thing', 0.007), ('caterpillar', 0.007), ('see', 0.007), ('time', 0.006),
# ('could', 0.006), ('think', 0.006), ('way', 0.006), ('would', 0.006), ('well', 0.005), ('went', 0.005),
# ('dear', 0.005), ('must', 0.005)]
