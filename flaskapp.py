import difflib
from flask import Flask
import pickle


df = pickle.load(open('df.pkl', 'rb'))
destinations = pickle.load(open('destinations.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World 123'


@app.route('/Recommendation/<string:desti_name>')
def Recommendations(desti_name):
    # Recommendation/Lahore (need to pass in this form to get a response)
    find_closed_match = difflib.get_close_matches(desti_name, destinations)
    closed_match = find_closed_match[0]
    index_of_desti = df[df['_key'] == closed_match].index.values[0]
    similarity_score = list(enumerate(similarity[index_of_desti]))
    sorted_desti_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    print("Destination Suggested to you are : ")
    name = []
    for desti in sorted_desti_list:
        index = desti[0]
        name.append(df.iloc[index]['_key'])

    print("Top 5 similar Items:")
    names = []
    for i in range(1, 6):
        print(i, ": ", name[i])
        names.append(name[i])

    return names


# main driver function
if __name__ == '__main__':
    app.run()
