# pylint: disable=unused-import, wrong-import-order
import neuralmonkey.checkpython
# pylint: enable=unused-import, wrong-import-order

import argparse
import os
import json
import datetime

import flask
from flask import Flask, request, Response, render_template
import numpy as np

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.dataset import Dataset, BatchingScheme
from neuralmonkey.experiment import Experiment


APP = Flask(__name__)
APP.config.from_object(__name__)
APP.config["experiment"] = None


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    src = os.path.join(root_dir(), filename)
    return open(src).read()


def run(data):  # pragma: no cover
    exp = APP.config["experiment"]
    dataset = Dataset(
        "request", data, BatchingScheme(batch_size=1), {},
        preprocessors=APP.config["preprocess"])

    _, response_data, _ = exp.run_model(dataset)

    return response_data


@APP.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        source_text = request.form["source"]
        data = {"source": [source_text.split()]}
        translation_response = run(data)
        translation = " ".join(translation_response["target"][0])
    else:
        source_text = "enter tokenized soruce language text here ."
        translation = ""

    return render_template(
        "server.html", translation=translation, source=source_text)


@APP.route("/run", methods=["POST"])
def post_request():
    start_time = datetime.datetime.now()
    request_data = request.get_json()

    if request_data is None:
        response_data = {"error": "No data were provided."}
        code = 400
    else:
        try:
            response_data = run(request_data)
            code = 200
        # pylint: disable=broad-except
        except Exception as exc:
            response_data = {"error": str(exc)}
            code = 400

    # take care of tensors returned by tensor runner
    for key, value in response_data.items():
        if isinstance(value[0], dict):
            new_value = [
                {k: v.tolist() for k, v in val.items()} for val in value]
            response_data[key] = new_value
        if isinstance(value[0], np.ndarray):
            response_data[key] = [x.tolist() for x in value]

    response_data["duration"] = (
        datetime.datetime.now() - start_time).total_seconds()
    json_response = json.dumps(response_data)
    response = flask.Response(json_response,
                              content_type="application/json; charset=utf-8")
    response.headers.add("content-length", len(json_response.encode("utf-8")))
    response.status_code = code
    return response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs Neural Monkey as a web server.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--configuration", type=str, required=True)
    parser.add_argument("--preprocess", type=str,
                        required=False, default=None)
    args = parser.parse_args()

    print("")

    if args.preprocess is not None:
        preprocessing = Configuration()
        preprocessing.add_argument("preprocess")
        preprocessing.load_file(args.preprocess)
        preprocessing.build_model()
        APP.config["preprocess"] = preprocessing.model.preprocess
    else:
        APP.config["preprocess"] = []

    exp = Experiment(config_path=args.configuration)
    exp.build_model()
    APP.config["experiment"] = exp
    APP.run(port=args.port, host=args.host)
