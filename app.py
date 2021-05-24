import flask
from flask import Flask
from flask import request, jsonify

from source.models.disorders import Disorders
from source.models.fundus import Fundus
from source.models.diseases import Disease
from source.models.infections import Infection

app = Flask(__name__)

fundus = Fundus()
diseases = Disease()
disorders = Disorders()
infections = Infection()


@app.route('/check_conn', methods=['GET'])
def check_conn():
    json_file = {'connection': 'connected', 'status': True}
    encoded_json = jsonify(json_file)
    return encoded_json


@app.route('/fundus', methods=['POST'])
def fundus_analysis():

    try:
        image = request.files.get('image', '')
        return fundus.prediction(image)
    except Exception as err:
        print(err)

    return fundus.json_file


@app.route('/disorders', methods=['POST'])
def disorders_analysis():

    try:
        image = request.files.get('image', '')
        return disorders.prediction(image)
    except Exception as err:
        print(err)

    return disorders.json_file


@app.route('/diseases', methods=['POST'])
def disease_analysis():

    try:
        image = request.files.get('image', '')
        return diseases.check_disease(image)
    except Exception as err:
        print(err)

    return diseases.json_file


@app.route('/infections', methods=['POST'])
def infection_analysis():

    try:
        image = request.files.get('image', '')
        return infections.check_infection(image)
    except Exception as err:
        print(err)

    return infections.json_file


if __name__ == '__main__':
    app.run()
