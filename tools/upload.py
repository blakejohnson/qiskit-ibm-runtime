#!/usr/bin/env python

import sys
import yaml
import requests


def make_request(url, func, **kwargs):
    try:
        response = func(url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as ex:
        if ex.response is not None:
            print("%s: %s: %s" % (str(ex), ex.response.text, url))
        raise


def upload(env, program_text_file, program_json_file):
    # parse config
    config = {}
    with open("config.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    endpoints = config.get(env)
    if endpoints is None:
        raise Exception(f"No {env} defined in the configuration")

    with open(program_text_file, "r") as f:
        data = f.read()
    program_text = base64.b64encode(data.encode("utf-8")).decode("utf-8")

    with open(program_json_file, "r") as f:
        metadata = json.loads(f.read())

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
