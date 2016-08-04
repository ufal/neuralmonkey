import argparse
import json
import datetime

import flask
from flask import Flask, request

from neuralmonkey.dataset import Dataset
from neuralmonkey.learning_utils import run_on_dataset
from neuralmonkey.checking import check_dataset_and_coders
from neuralmonkey.run import initialize_for_running

APP = Flask(__name__)
APP.config.from_object(__name__)
APP.config['args'] = None
APP.config['sess'] = None

@APP.route('/', methods=['GET', 'POST'])
def post_request():
    start_time = datetime.datetime.now()
    request_data = request.get_json()

    if request_data is None:
        response_data = {"error": "No data were provided."}
        code = 400
    else:
        args = APP.config['args']
        sess = APP.config['sess']

        try:
            dataset = Dataset("request", request_data, {})
            check_dataset_and_coders(dataset, args.encoders)

            result, _, _ = run_on_dataset(
                sess, args.runner, args.encoders + [args.decoder], args.decoder,
                dataset, args.evaluation, args.postprocess, write_out=True)
            response_data = {args.decoder.data_id: result}
            code = 200
        #pylint: disable=broad-except
        except Exception as exc:
            response_data = {'error': str(exc)}
            code = 400

    response_data['duration'] = (datetime.datetime.now() - start_time).total_seconds()
    json_response = json.dumps(response_data)
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = code
    return response

def main():
    parser = argparse.ArgumentParser(description="Runs Neural Monkey as a web server.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--configuration", type=str)
    cli_args = parser.parse_args()

    print("")

    args, sess = initialize_for_running(cli_args.configuration)
    APP.config['args'] = args
    APP.config['sess'] = sess
    APP.run(port=cli_args.port)
