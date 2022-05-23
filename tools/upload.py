# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Upload programs."""

import os
import base64
import sys
import json
import requests
import yaml

# pylint: disable=unspecified-encoding


def make_request(url, func, **kwargs):
    """Generic fuction to make any http request."""
    try:
        response = func(url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as ex:
        if ex.response is not None:
            print("%s: %s: %s" % (str(ex), ex.response.text, url))
        raise


def upload(env, program_text_file, program_json_file):
    """Upload progam data and metadata."""
    # parse config
    config = {}
    with open("config.yaml", "r") as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    endpoints = config.get(env)
    if endpoints is None:
        raise Exception(f"No {env} defined in the configuration")

    with open(program_text_file, "r") as data_file:
        data = data_file.read()
    program_text = base64.b64encode(data.encode("utf-8")).decode("utf-8")

    with open(program_json_file, "r") as metadata_file:
        metadata = json.loads(metadata_file.read())

    metadata["data"] = program_text
    metadata["is_public"] = True
    metadata["program_id"] = program_text_file.split(".")[0]

    print(f"Uploading to {env}")
    for endpoint in endpoints:
        url = endpoint["url"] + "/programs"
        token = os.getenv(endpoint["token_key"])
        assert token != "", endpoint["token_key"]
        print(url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        make_request(url, requests.post, headers=headers.copy(), json=metadata)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <environment> <program file> <program json file>")
        sys.exit(1)

    upload(sys.argv[1], sys.argv[2], sys.argv[3])
