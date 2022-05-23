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

"""Update programs."""

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


# Only doing updates. For adds and deletes, those are usually one time deals that
# can be handled separately. Keeps this code simple for what we need 99% of the time.
def update(envs=None):
    """Update progam data and metadata."""
    if envs is None:
        envs = ["staging"]

    # parse config
    config = {}
    with open("config.yaml", "r") as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    for env in envs:
        endpoints = config.get(env)
        if endpoints is None:
            raise Exception(f"No {env} defined in the configuration")

        print(f"Updating {env}:")
        for endpoint in endpoints:
            url = endpoint["url"]
            token = os.getenv(endpoint["token_key"])
            assert token != "", endpoint["token_key"]
            print(url)
            for program in endpoint["programs"]:
                print(f"- {program}")
                program_url = f"{url}/programs/{program}"
                program_name = program.replace("-", "_")

                # patch with metadata
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                with open(f"programs/{program_name}.json", "r") as metadata_file:
                    metadata = metadata_file.read()
                make_request(
                    program_url,
                    requests.patch,
                    headers=headers.copy(),
                    json=json.loads(metadata),
                )

                # update data
                headers["Content-Type"] = "application/octet-stream"
                with open(f"programs/{program_name}.py", "r") as data_file:
                    data = data_file.read()
                program_text = base64.b64encode(data.encode("utf-8")).decode("utf-8")
                make_request(
                    program_url + "/data",
                    requests.put,
                    headers=headers.copy(),
                    data=program_text,
                )

                # update public
                del headers["Content-Type"]
                make_request(program_url + "/public", requests.put, headers=headers.copy())


if __name__ == "__main__":
    update(sys.argv[1:])
