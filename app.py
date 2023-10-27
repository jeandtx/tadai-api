from flask import Flask, jsonify, request
from models.attributes import get_recommendations_attributes
from main import merged_model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    data = {
        'message': 'Hello, World!',
        'version': 1.2,
        'status': 'success',
    }
    response = jsonify(data)
    response.status_code = 200  # Set the status code to 200 (OK)
    return response

@app.route('/attributes', methods=['GET'])
def get_attributes():
    song_title = request.args.get('title')
    song_artist = request.args.get('artist')
    nb_of_recommendations = request.args.get('nb_of_recommendations', default=10, type=int)

    if not song_title or not song_artist:
        return jsonify({'error': 'Both title and artist parameters are required',
                        'example': 'http://api/attributes?title=YourSongTitle&artist=YourSongArtist&nb_of_recommendations=5'
                        }), 400

    recommendations = get_recommendations_attributes(song_title, song_artist, number_of_songs=nb_of_recommendations)[['Name', 'Artist']]
    data = {
        'message': 'Please find the recommendations in the data field',
        'status': 'success',
        'data': recommendations.to_json(orient='records')
    }
    response = jsonify(data)
    response.status_code = 200
    return response


@app.route('/predict', methods=['GET'])
def get_predict():
    song_title = request.args.get('title')
    song_artist = request.args.get('artist')
    nb_of_recommendations = request.args.get('nb_of_recommendations', default=10, type=int)

    if not song_title or not song_artist:
        return jsonify({'error': 'Both title and artist parameters are required',
                        'example': 'http://13.38.95.183/attributes?title=Big%20City%20life&artist=Mattafix&nb_of_recommendations=1'
                        }), 400

    recommendations = merged_model(song_title, song_artist, number_of_songs=nb_of_recommendations)[['Name', 'Artist']]
    data = {
        'message': 'Please find the recommendations in the data field',
        'status': 'success',
        'data': recommendations.to_json(orient='records')
    }
    response = jsonify(data)
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run()
