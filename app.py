import flask
from flask import Flask
from flask import request, jsonify

from source.model_classes.fundus import Fundus

app = Flask(__name__)


@app.route('/check_conn', methods=['GET'])
def check_conn():
    json_file = {}
    json_file['connection'] = 'connected'
    json_file['status'] = True

    encoded_json = jsonify(json_file)
    return encoded_json


@app.route('/fundus', methods=['POST'])
def fundus_analysis():

    fundus = Fundus()

    try:
        image = request.files.get('image', '')
        return fundus.prediction(image)
    except Exception as err:
        print(err)

    return fundus.json_file


if __name__ == '__main__':
    app.run()
