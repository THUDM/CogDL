#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import sys
Default_BACKEND = 'torch'
# BACKEND = 'torch'

# Check for backend.json files
cogdl_backend_dir = os.path.expanduser('~')
if not os.access(cogdl_backend_dir, os.W_OK):
    cogdl_backend_dir = '/tmp'
cogdl_dir = os.path.join(cogdl_backend_dir, '.cogdl')

config = {
    'backend': Default_BACKEND,
}
if not os.path.exists(cogdl_dir):
    path = os.path.join(cogdl_dir, 'cogdl_backend.json')
    os.makedirs(cogdl_dir)
    with open(path, "w") as f:
        json.dump(config, f)
    BACKEND = config['backend']
    sys.stderr.write("Create the backend configuration file :" + path + '\n')
else:
    path = os.path.join(cogdl_dir, 'cogdl_backend.json')
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
        BACKEND = load_dict['backend']

# Set backend based on BACKEND.
if 'CogDLBACKEND' in os.environ:
    backend = os.environ['CogDLBACKEND']
    if backend in ['jittor', 'torch']:
        BACKEND = backend
    else:
        print("CogDL backend not selected or invalid.  "
                "Assuming PyTorch for now.")
        path = os.path.join(cogdl_dir, 'cogdl_backend.json')
        with open(path, "w") as f:
            json.dump(config, f)

def backend():
    return BACKEND