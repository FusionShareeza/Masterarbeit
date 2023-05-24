from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from backend import*
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/results_tenant', methods=['GET'])
def get_results():
    results = get_improvement_results()
    return jsonify(results)

@app.route('/json_saved', methods=['GET'])
def get_from_json():
    results = get_results_from_json()
    return jsonify(results)

@app.route('/json_saved_debitor', methods=['GET'])
def get_from_json_debitor():
    results = get_results_from_json_debitor()
    return jsonify(results)

@app.route('/aktueller_zustand', methods=['GET'])
def get_current_status_func():
    results = get_current_status()
    return jsonify(results)

@app.route('/mandantenerkennung', methods=['GET'])
def get_debitor_results_func():
    results = get_debitor_results()
    return jsonify(results)


if __name__ == '__main__':
   app.run(port=5002)