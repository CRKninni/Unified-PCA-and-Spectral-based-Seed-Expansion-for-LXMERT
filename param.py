# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
# import random

# import numpy as np
# import torch

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-path', type = str,
                        default='',
                        help='')
    
    parser.add_argument('--question', type = str,
                        default = '',
                        help = '')
    
    parser.add_argument('--method-name', type = str,
                        default = 'dsm_grad_cam',
                        help = '')

    # Parse the arguments.
    args = parser.parse_args()

    return args


args = parse_args()
