import difflib
from flask import Flask
import gensim.downloader as api
import numpy as np
import pickle

clf = pickle.load(open('gradientBoosting.pkl', 'rb'))
wv = api.load('word2vec-google-news-300')

df = pickle.load(open('df.pkl', 'rb'))
destinations = pickle.load(open('destinations.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello In the World of Digital Cloud'


@app.route('/Recommendation/<string:desti_name>')
def Recommendations(desti_name):
    # Recommendation/Lahore (need to pass in this form to get a response)
    find_closed_match = difflib.get_close_matches(desti_name, destinations)
    closed_match = find_closed_match[0]
    index_of_desti = df[df['_key'] == closed_match].index.values[0]
    similarity_score = list(enumerate(similarity[index_of_desti]))
    sorted_desti_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # print("Destination Suggested to you are : ")
    name = []
    for desti in sorted_desti_list:
        index = desti[0]
        name.append(df.iloc[index]['_key'])

    # print("Top 5 similar Items:")
    names = []
    for i in range(1, 6):
        # print(i, ": ", name[i])
        names.append(name[i])

    return names


@app.route('/Scoring<string/<string:comments>')
def Scoring(comments):
    comments_list = comments.split('///')
    scoring_comments = []
    for comment in comments_list:
        scoring_comments.append(predict(comment))

    n = 0
    for i in range(len(scoring_comments)):
        if scoring_comments[i] >= 3:
            n = n + 1

    print(n)
    print(scoring_comments)
    percentage = (n / len(scoring_comments)) * 100
    return int(percentage)


def predict(comment):
    token = sent_vec(comment)
    x2d = [np.stack(token)]
    y = clf.predict(x2d)
    # print(y)
    x = str(y[0])
    return x


def sent_vec(sent):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res / ctr
    return wv_res


# main driver function
if __name__ == '__main__':
    app.run()
