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
    print(results)
    return jsonify(results)

@app.route('/json_saved', methods=['GET'])
def get_from_json():
    results = get_results_from_json()
    return jsonify(results)

@app.route('/json_saved_debitor', methods=['GET'])
def get_from_json_debitor():
    results = get_results_from_json_debitor()
    print(results)
    return jsonify(results)

@app.route('/mandantenerkennung', methods=['GET'])
def get_debitor_results_func():
    results = get_debitor_results()
    return jsonify(results)

@app.route('/lieferantenerkennung', methods=['GET'])
def get_vendor_results_func():
    results = get_vendor_results()
    return jsonify(results)

@app.route('/positionserkennung', methods=['GET'])
def get_pos_results_func():
    results = get_pos_results()
    return jsonify(results)

@app.route('/smartinvoice', methods=['GET'])
def get_smart_invoice_error_fun():
    results = get_smart_invoice_error()
    print(results)
    return jsonify(results)

@app.route('/aktueller_zustand', methods=['GET'])
def get_current_status_func():
    results = get_current_status()
    return jsonify(results)

@app.route('/komplette_ergebnisse', methods=['GET'])
def get_results_from_json_complete_func():
    results = get_results_from_json_complete()
    return jsonify(results)

@app.route('/komplette_ergebnisse_frequenz', methods=['GET'])
def get_results_from_json_complete_frequency_func():
    results = get_results_from_json_complete_frequency()
    return jsonify(results)


@app.route('/sollwerte', methods=['GET'])
def get_sollwerte_func():
    results = get_sollwerte()
    return jsonify(results)

@app.route('/sollwerte_verbesserungen', methods=['GET'])
def get_sollwerte_verbesserungen_func():
    results = get_commments_from_sollwerte()
    return jsonify(results)

@app.route('/autotrain_info', methods=['GET'])
def get_autotrain_results_func():
    results = get_autotrain_results()
    return results


if __name__ == '__main__':
   app.run(port=5002)