#!/usr/bin/env python3
"""Loads and builds a given config file in memory.

Can be used for checking that a model can be loaded successfully, or for
generating a vocabulary from a dataset, without the need to run the model.
"""

import argparse
import collections
from typing import Any, Dict, cast

import neuralmonkey
from neuralmonkey.config.parsing import parse_file
from neuralmonkey.config.builder import build_config, ObjectRef, ClassSymbol


def _patch_config_builder():
    imports = set()
    statements = []
    deps = []
    obj_stack = []

    def get_class_name(symbol: ClassSymbol):
        name = symbol.clazz
        if name.startswith("tf."):
            return name
        full_name = "neuralmonkey." + name
        module, _, _ = full_name.rpartition(".")
        imports.add("import " + module)
        return full_name

    def build_object(value: str,
                     all_dicts: Dict[str, Any],
                     existing_objects: Dict[str, Any],
                     depth: int) -> Any:
        if depth > 20:
            raise AssertionError(
                "Config recursion should not be deeper that 20.")

        if (isinstance(value, collections.Iterable) and
            not isinstance(value, str)):
            objects = [build_object(
                val, all_dicts, existing_objects, depth + 1) for val in value]
            if isinstance(value, tuple):
                if len(objects) == 1:
                    objects[0] += ","  # Singleton tuple needs a comma.
                return "(" + ", ".join(objects) + ")"
            else:
                return "[" + ", ".join(objects) + "]"

        if isinstance(value, ObjectRef):
            if obj_stack:
                deps.append((obj_stack[-1], value.name))

            if value.name not in existing_objects:
                obj_stack.append(value.name)
                clazz = all_dicts[value.name]["class"]
                args = [
                    "\n    {}={}".format(key, build_object(
                        val, all_dicts, existing_objects, depth + 1))
                    for key, val in all_dicts[value.name].items()
                    if key != "class"
                ]
                obj_stack.pop()
                statements.append(
                    "{} = {}({})".format(
                        value.name, get_class_name(clazz), ",".join(args)))

                existing_objects[value.name] = True
            return value.expression

        if isinstance(value, ClassSymbol):
            return get_class_name(value)

        return repr(value)

    neuralmonkey.config.builder.build_object = build_object

    return imports, statements, deps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="a configuration file")
    result_group = parser.add_mutually_exclusive_group()
    result_group.add_argument("--code", "-c", action="store_const",
                              dest="result_type", const="code",
                              help="instead of building the config, generate "
                              "equivalent Python code and write it to stdout")
    result_group.add_argument("--graph", "-g", action="store_const",
                              dest="result_type", const="graph",
                              help="instead of building the config, generate "
                              "a dependency graph in the DOT language and "
                              "write it to stdout")
    parser.set_defaults(result_type="model")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        _, config_dict = parse_file(f)

    if args.result_type in ["code", "graph"]:
        imports, statements, deps = _patch_config_builder()

    config, _ = build_config(config_dict, ignore_names=set())

    if args.result_type == "code":
        print("import argparse\nimport tensorflow as tf")
        print(*sorted(imports), sep="\n", end="\n\n")
        print(*statements, sep="\n", end="\n\n")
        print("model = argparse.Namespace({})".format(
            ",".join("\n    {}={}".format(key, config[key]) for key in config)
        ))
    elif args.result_type == "graph":
        print("digraph model {")
        for name, section in config_dict.items():
            if "class" in section:
                print('{0} [label="{0}\n{1}"];'
                    .format(name, section["class"].create().__name__))
        printed_deps = set()
        for src, tgt in deps:
            if (src, tgt) not in printed_deps:
                print('{} -> {};'.format(tgt, src))
                printed_deps.add((src, tgt))
        print("}")

if __name__ == "__main__":
    main()
