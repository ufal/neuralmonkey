import argparse
import os
import json
from ansi2html import Ansi2HTMLConverter
from flask import Flask, Response

APP = Flask(__name__)
APP.config.from_object(__name__)
APP.config['log_dir'] = None

ANSI_CONV = Ansi2HTMLConverter()

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    src = os.path.join(root_dir(), filename)
    return open(src).read()

@APP.route('/', methods=['GET'])
def index():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")

@APP.route('/experiments', methods=['GET'])
def list_experiments():
    log_dir = APP.config['log_dir']
    experiment_list = \
        [dr for dr in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, dr))]
    json_response = json.dumps({'experiments': experiment_list}).decode('unicode-escape')

    response = Response(json_response,
                        content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response))
    response.status_code = 200
    return response

@APP.route('/experiments/<path:path>', methods=['GET'])
def get_experiment(path):
    log_dir = APP.config['log_dir']
    complete_path = os.path.join(log_dir, path)
    if os.path.isfile(complete_path):
        file_content = get_file(complete_path)
        if path.endswith(".log"):
            result = ANSI_CONV.convert(file_content, full=False)
        elif path.endswith(".ini"):
            result = file_content
        else:
            result = "Unknow file type: \"{}\".".format(complete_path)
    else:
        result = "File \"{}\" does not exist.".format(complete_path)
    return Response(result, mimetype='text/html', status=200)

@APP.route('/', defaults={'path': ''})
@APP.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    try:
        complete_path = os.path.join(root_dir(), path)
        ext = os.path.splitext(path)[1]
        mimetype = mimetypes.get(ext, "text/html")
        content = get_file(complete_path)
        return Response(content, mimetype=mimetype)
    except IOError:
        return Response("'{}' not found.".format(path), status=404)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description="Runs the Experiment LogBook server")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-dir", type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print "The log directory '{}' does not exist."
        exit(1)

    APP.config['log_dir'] = args.log_dir

    APP.run(port=args.port)
