import argparse
import os
import json
import datetime

import flask
from flask import Flask, request, Response, render_template

from neuralmonkey.dataset import Dataset
from neuralmonkey.learning_utils import run_on_dataset
from neuralmonkey.run import CONFIG, initialize_for_running


APP = Flask(__name__)
APP.config.from_object(__name__)
APP.config["args"] = None


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    src = os.path.join(root_dir(), filename)
    return open(src).read()


def translate(data):  # pragma: no cover
    args = APP.config["args"]
    dataset = Dataset("request", data, {})
    # TODO check the dataset
    # check_dataset_and_coders(dataset, args.encoders)

    _, response_data = run_on_dataset(
        args.tf_manager, args.runners,
        dataset, args.postprocess, write_out=False)

    return response_data


@APP.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        source_text = request.form["source"]
        data = {"source": [source_text.split()]}
        translation_response = translate(data)
        translation = " ".join(translation_response["target"][0])
    else:
        source_text = "enter tokenized soruce language text here ."
        translation = ""

    return render_template(
        "server.html", translation=translation, source=source_text)


@APP.route("/translate", methods=["POST"])
def post_request():
    start_time = datetime.datetime.now()
    request_data = request.get_json()

    if request_data is None:
        response_data = {"error": "No data were provided."}
        code = 400
    else:
        try:
            response_data = translate(request_data)
            code = 200
        # pylint: disable=broad-except
        except Exception as exc:
            response_data = {"error": str(exc)}
            code = 400

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
    cli_args = parser.parse_args()

    print("")

    # pylint: disable=no-member
    CONFIG.load_file(cli_args.configuration)
    CONFIG.build_model()
    initialize_for_running(CONFIG.model.output, CONFIG.model.tf_manager, None)
    APP.config["args"] = CONFIG.model
    APP.run(port=cli_args.port, host=cli_args.host)
