import os
import json
import sys
import pprint
import re
import argparse
import warnings
from collections import defaultdict

import yaml
import math
import easydict
from .utils import Capturing


def extend_config(cfg_path, child = None):
    if not os.path.exists(cfg_path):
        warnings.warn(f'::: File {cfg_path} was not found!')
        return child

    with open(cfg_path, 'rt', encoding="utf8") as fd:
        parent_cfg = yaml.load(fd, Loader = yaml.FullLoader)

    if child is not None:
        parent_cfg.update(child)

    if '$extends$' in parent_cfg:
        path = parent_cfg['$extends$']
        del parent_cfg['$extends$']
        parent_cfg = extend_config(child = parent_cfg, cfg_path = path)

    return parent_cfg

def load_args(args):
    cfg = extend_config(cfg_path = f'{args.config_file}', child = None)

    if cfg is None:
        return args, cfg

    if '$includes$' in cfg:
        included_cfg = {}
        for included_cfg_path in cfg['$includes$']:
            included_cfg = {
                **included_cfg,
                ** extend_config(cfg_path = included_cfg_path, child = None)
            }

        cfg = {**cfg, **included_cfg}
        del cfg['$includes$']

    for key, value in cfg.items():
        if key in args and args.__dict__[key] is not None:
            continue

        if not isinstance(args, dict):
            args.__dict__[key] = value
        else:
            args[key] = value

    return args, cfg

def flatten_dict_keys(original_key, values):
    new_values = []
    for key, value in values.items():
        new_key = original_key + '.' + key
        if isinstance(value, list):
            new_values.append((new_key, value, type(value)))
            continue

        if isinstance(value, dict):
            extra_new_values = flatten_dict_keys(new_key, value)
            new_values.extend(extra_new_values)

        if not isinstance(value, dict):
            new_values.append((new_key, value, type(value)))

    return new_values

def unflatten_dict_keys(dict_args, args):
    dict_values = dict_args.copy()

    for key, value in args.items():
        if '.' in key:
            continue

        if key in dict_values:
            dict_values[key] = {**dict_values[key], **value}
        else:
            dict_values[key] = value

    unnested = defaultdict(dict)
    for key, value in args.items():
        if '.' not in key:
            continue

        root = '.'.join(key.split('.')[:-1])
        value = {key.split('.')[-1]: value}
        unnested[root].update(value)

    if len(unnested):
        dict_values = unflatten_dict_keys(dict_values.copy(), unnested)

    return dict_values

def update_parser(parser, args):
    for key, value in args.__dict__.items():
        if isinstance(value, dict):
            new_values = flatten_dict_keys(key, value)

            for arg_name, default_value, arg_type in new_values:
                parser.add_argument(
                    f'--{arg_name}',
                    type = arg_type,
                    default = default_value,
                    required = False
                )

            continue

        if isinstance(value, list) and len(value) > 0:
            parser.add_argument(
                f'--{key}',
                type = type(value[0]),
                default = value,
                nargs = '+',
                required = False
            )
            continue

        if key == 'config_file':
            continue

        parser.add_argument(f'--{key}', type = type(value), default = value)

    return parser

def instantiate_references(flattened_args):
    refs = {}
    for name, value in flattened_args.items():
        if not isinstance(value, str):
            continue

        if value.startswith('${') and value.endswith('}'):
            refs[name] = flattened_args[value[2:-1]]

    for name, value in refs.items():
        flattened_args[name] = value

    return flattened_args

def compute_expressions(flattened_args):
    variable_re = re.compile(r'\${(.*?)}')

    new_values = {}
    for name, value in flattened_args.items():
        if not isinstance(value, str):
            continue

        if value.startswith('`') and value.endswith('`'):
            expression_str = value[1:-1]
            # replace all the references with appropriate values from flattened_args
            for ref in variable_re.findall(expression_str):
                expression_str = expression_str.replace(f'${{{ref}}}', str(flattened_args[ref]))

            new_values[name] = eval(
                expression_str,
                {'__builtins__': None},
                {'sqrt': lambda x: x ** 0.5, 'pow': lambda x, y: x ** y, 'log': math.log}
            )

    for name, value in new_values.items():
        flattened_args[name] = value

    return flattened_args

def find_config_file():
    config_path = None
    is_resume = False
    has_equals = False
    for i in range(len(sys.argv)):
        if '--config_file' in sys.argv[i] or '--resume_from' in sys.argv[i]:
            if '=' in sys.argv[i]:
                config_path = sys.argv[i].split('=')[-1]
                has_equals = True
            else:
                config_path = sys.argv[i + 1] if len(sys.argv) > i + 1  else None

            if '--resume_from' in sys.argv[i]:
                is_resume = True
            break

    if config_path is None:
        return None

    # removing both key and value
    sys.argv.pop(i)
    if not has_equals:
        sys.argv.pop(i)

    if is_resume:
        config_path = f'{config_path}/config.json'

    return config_path

def define_args(extra_args = None, verbose = True, require_config_file = False, print_fn = print):
    config_path = find_config_file()

    if require_config_file and config_path is None:
        raise Exception('No config file provided!')

    is_resume = False
    if config_path is not None:
        if config_path.endswith('.json'):
            is_resume = True

            # Read the config file and update the args.
            with open(config_path, 'r') as f:
                config = easydict.EasyDict(json.load(f))

            # strip the last part of the path ("config.json')")
            config.resume_from = config_path[:-len('config.json')]
            cfg_args = easydict.EasyDict(config.copy())
        else:
            cfg_args, _ = load_args(argparse.Namespace(config_file = config_path))
    else:
        cfg_args = argparse.Namespace(_={})

    parser = argparse.ArgumentParser(description='Do stuff.')

    if not is_resume:
        parser.add_argument('--name', type = str, default = 'test')
        parser.add_argument('--group', type = str, default = 'default')
        parser.add_argument('--notes', type = str, default = '')
        parser.add_argument("--mode", type = str, default = 'run')
        parser.add_argument("--debug", type = int, default = 0)

    if extra_args is not None:
        for name, arguments in extra_args:
            parser.add_argument(name, **arguments)

    # Needed to be able to update nested config keys
    # Obviously (?) dosen't work with list arguments (such as model heads and losses).
    parser = update_parser(parser = parser, args = cfg_args)
    flattened_args = parser.parse_args()
    flattened_args = flattened_args.__dict__

    flattened_args = instantiate_references(flattened_args)
    flattened_args = compute_expressions(flattened_args)

    # Make an EasyDict with all the args. This is used in all the main actors.
    nested_args = unflatten_dict_keys({}, flattened_args)
    args = easydict.EasyDict(nested_args)

    if os.path.exists('configs/env_config.yaml') and 'env' in args:
        with open('configs/env_config.yaml', 'rt', encoding="utf8") as fd:
            env_cfg = yaml.load(fd, Loader = yaml.FullLoader)

        if args.env not in env_cfg:
            raise Exception(f'{args.env} not found in env_config.yaml! Configured environments: {list(env_cfg.keys())}')

        args.environment = env_cfg[args.env]

    output_string = ""

    if args.debug:
        output_string =  "#############################\n"
        output_string += '########🐞DEBUG MODE🐞########\n'
        output_string += '#############################\n'
        print_fn(output_string)

        print_fn('[🐞DEBUG MODE🐞] Overriding seed argument...')
        args.seed = 69

        print_fn("[🐞DEBUG MODE🐞] Changing name to:", args.name + '-DEBUG')
        args.name = args.name + '-DEBUG'

        print_fn("[🐞DEBUG MODE🐞] Changing WANDB_MODE to 'dryrun'",)
        args.mode = 'dryrun'

    if verbose:
        with Capturing() as output:
            pprint.pprint(args)
        for o in output: print_fn(o)

    return args