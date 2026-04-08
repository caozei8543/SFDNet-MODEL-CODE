"""
训练参数解析
"""

import yaml
import argparse


def parse_train_options():
    parser = argparse.ArgumentParser(
        description='SFDNet Training Options'
    )
    parser.add_argument(
        '--config', type=str, default='configs/train_lol.yaml',
        help='Path to the configuration YAML file.'
    )
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='GPU device id.'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from.'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['gpu'] = args.gpu
    config['resume'] = args.resume

    return config