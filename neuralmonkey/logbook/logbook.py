import argparse
import os
import html
import json
from flask import Flask, Response
from pygments import highlight
from pygments.lexers.configs import IniLexer
from pygments.formatters import HtmlFormatter
import ansiconv


APP = Flask(__name__)
APP.config.from_object(__name__)
APP.config['logdir'] = None


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
    logdir = APP.config['logdir']
    experiment_list = [dr for dr in os.listdir(logdir)
                       if os.path.isdir(os.path.join(logdir, dr))
                       and os.path.isfile(os.path.join(
                           logdir, dr, 'experiment.ini'))]

    if os.path.isfile(os.path.join(logdir, 'experiment.ini')):
        experiment_list.append(".")

    json_response = json.dumps({'experiments': experiment_list})

    response = Response(json_response,
                        content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response))
    response.status_code = 200
    return response


@APP.route('/experiments/<path:path>', methods=['GET'])
def get_experiment(path):
    logdir = APP.config['logdir']
    complete_path = os.path.join(logdir, path)
    if os.path.isfile(complete_path):
        file_content = get_file(complete_path)
        if path.endswith(".log"):
            result = ansiconv.to_html(html.escape(file_content))
        elif path.endswith(".ini"):
            lexer = IniLexer()
            formatter = HtmlFormatter(linenos=True)
            result = highlight(file_content, lexer, formatter)
        else:
            result = "Unknown file type: '{}'.".format(complete_path)
    else:
        result = "File '{}' does not exist.".format(complete_path)
    return Response(result, mimetype='text/html', status=200)


@APP.route("/ansiconv.css")
def get_ansiconv_css():
    return ansiconv.base_css()


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


def main():
    parser = argparse.ArgumentParser(
        description="Runs the Experiment LogBook server")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)

    if not os.path.isdir(logdir):
        print("The log directory '{}' does not exist.")
        exit(1)

    APP.config['logdir'] = logdir

    APP.run(port=args.port, host=args.host)


if __name__ == '__main__':
    main()
