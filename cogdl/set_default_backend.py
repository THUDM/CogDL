import argparse
import os
import json


def set_default_backend(default_dir, backend_name):
    os.makedirs(default_dir, exist_ok=True)
    config_path = os.path.join(default_dir, "cogdl_backend.json")
    with open(config_path, "w") as config_file:
        json.dump({"backend": backend_name.lower()}, config_file)
    print(
        'Setting the default backend to "{}". You can change it in the '
        "~/.cogdl/config.json file or export the CogDLBACKEND environment variable.  "
        "Valid options are: pytorch, jittor (all lowercase)".format(backend_name)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_dir", type=str, default=os.path.join(os.path.expanduser("~"), ".cogdl"))
    parser.add_argument("backend", nargs=1, type=str, choices=["torch", "jittor"], help="Set default backend")
    args = parser.parse_args()
    set_default_backend(args.default_dir, args.backend[0])
