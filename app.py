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





if __name__ == '__main__':
    app.run()
