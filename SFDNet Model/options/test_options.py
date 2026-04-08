"""
测试参数解析
"""

import yaml
import argparse


def parse_test_options():
    parser = argparse.ArgumentParser(
        description='SFDNet Test Options'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the configuration YAML file.'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the pretrained model checkpoint.'
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Path to the input low-light images.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/',
        help='Path to save the enhanced images.'
    )
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['checkpoint'] = args.checkpoint
    config['input_dir'] = args.input_dir
    config['output_dir'] = args.output_dir
    config['gpu'] = args.gpu

    return config