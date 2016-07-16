import configparser
import re

LINE_NUM = re.compile(r"^(.*) ([0-9]+)$")

def parse_config(filename):
    line_numbers = (line.strip() + " " + str(i + 1)
                    if line.strip() != "" else ""
                    for i, line in
                    enumerate(open(filename).readlines())
                   )
    config = configparser.ConfigParser()
    config.read_file(line_numbers, source=filename)
    new_config = {}
    for section in config.sections():
        new_config[section] = {}
        for key in config[section]:
            m = LINE_NUM.match(config[section][key])
            new_config[section][key] = m.group(2), m.group(1)

    return new_config

config = parse_config('tests/small.ini')

for section in config:
    print("SECTION:", section)
    for key in config[section]:
        print(key, config[section][key])
