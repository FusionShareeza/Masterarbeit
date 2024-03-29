from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from backend import*
from flask_cors import CORS
from queue import Queue

app = Flask(__name__)
api_queue = Queue()
debitor_queue = Queue()
vendor_queue = Queue()
rechnungs_queue = Queue()


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
    debitor_queue.put(get_debitor_results)
    if debitor_queue.qsize() == 1:
        request_func = debitor_queue.get()
        results = request_func()
        return jsonify(results)
    if not debitor_queue.empty():
        request_func = debitor_queue.get()
        results = request_func()
        return jsonify(results)
    
@app.route('/lieferantenerkennung', methods=['GET'])
def get_vendor_results_func():
    vendor_queue.put(get_vendor_results)
    if vendor_queue.qsize() == 1:
        request_func = vendor_queue.get()
        results = request_func()
        return jsonify(results)
    if not vendor_queue.empty():
        request_func = vendor_queue.get()
        results = request_func()
        return jsonify(results)


@app.route('/positionserkennung', methods=['GET'])
def get_pos_results_func(): 
    rechnungs_queue.put(get_pos_results)
    if vendor_queue.qsize() == 1:
        request_func = rechnungs_queue.get()
        results = request_func()
        return jsonify(results)
    if not rechnungs_queue.empty():
        request_func = rechnungs_queue.get()
        results = request_func()
        return jsonify(results)

@app.route('/smartinvoice', methods=['GET'])
def get_smart_invoice_error_func():
    results = get_smart_invoice_error()
    print(results)
    return jsonify(results)

@app.route('/doubleiban', methods=['GET'])
def get_double_iban_error_func():
    results = get_double_iban_error()
    return jsonify(results)


@app.route('/smartinvoice_distribution', methods=['GET'])
def get_smart_invoice_error_distribution_func():
    results = get_smart_invoice_error_distribution()
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
    return jsonify(results)

@app.route('/results_table', methods=['GET'])
def get_results_table_func():
    api_queue.put(get_results_table)
    if api_queue.qsize() == 1:
        request_func = api_queue.get()
        results = request_func()
        return jsonify(results)
    if not api_queue.empty():
        request_func = api_queue.get()
        results = request_func()
        return jsonify(results)


@app.route('/pos_ven_results', methods=['GET'])
def get_pos_ven_debitor_results_func():
    results = get_pos_ven_debitor_results()
    print(results)
    return jsonify(results)

if __name__ == '__main__':
   app.run(port=5002)