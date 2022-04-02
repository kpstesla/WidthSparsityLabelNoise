"""
Filename: args_config.py
Author: Kyle Whitecross
Description: Contains methods to intelligently parse config files and commandline arguments.
"""

from argparse import ArgumentParser
import sys
import yaml

DEFAULT_CONFIG = "exps/template.yaml"


def parse_args_with_config(args_str=None):
    # try and parse config first
    parser = ArgumentParser(description="Dynamic config based argument parser")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    if args_str is None:
        args_str = sys.argv
    known_args, _ = parser.parse_known_args(args_str)

    # open config
    f = open(known_args.config)
    config = yaml.safe_load(f)
    f.close()

    # add config options to arg parser
    parser_config = ArgumentParser(description="Dynamic config based argument parser")
    parser_config.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    for key, val in config.items():
        if type(val) is str:
            val = val.lower()
        if type(val) is bool:
            parser_config.add_argument(f"--{key.lower()}", action='store_true', default=val)
            parser_config.add_argument(f"--disable_{key.lower()}", action="store_true", default=not val)
        else:
            parser_config.add_argument(f"--{key.lower()}", default=val)

    # parse arguments with new arg parser
    args = parser_config.parse_args(args_str)

    # use 'disable_' arguments to optionally disable boolean arguments
    for key, val in args.__dict__.items():
        if 'disable_' in key:
            suffix = key.split('disable_')[-1]
            if args.__dict__[suffix] and val:
                args.__dict__[suffix] = False
    return args
