# define the root path of the project

import os

def root_path():
    return os.path.dirname(os.path.abspath(__file__)) + '/'

def data_path():
    return root_path() + 'data/'

def data_input_path():
    return data_path() + 'input/'

def data_output_path():
    return data_path() + 'output/'

