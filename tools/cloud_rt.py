import json
import base64
import os
import logging
from types import SimpleNamespace

import requests

CLOUD_RUNTIME_URL_PROD = "https://us-east.quantum-computing.cloud.ibm.com"
CLOUD_RUNTIME_URL_STAGING = "https://us-east.quantum-computing.test.cloud.ibm.com"
LOG = logging.getLogger(__name__)


class CloudRuntimeClient:

    def __init__(self):
        url, token = self._get_url_token()
        self._base_url = url + "/programs"
        self._access_token = token
        self._header = {'Content-Type': 'application/json',
                        'Authorization': f"Bearer {self._access_token}"}

    def update_program_id(self, old_id, new_id):
        url = self._base_url + f"/{old_id}"
        self._make_request(url, requests.patch, json={"id": new_id})

    def update_program(self, program_id, data, metadata):
        url = self._base_url + f"/{program_id}/data"
        program_data = self._read_data(data)
        headers = {'Content-Type': 'application/octet-stream'}
        self._make_request(url, requests.put, data=program_data, headers=headers)

        url = self._base_url + f"/{program_id}"
        program_metadata = self._read_metadata(metadata)
        self._make_request(url, requests.patch, json=program_metadata)

    def upload_program(self, data, metadata):
        payload = self._read_metadata(metadata)
        payload["data"] = self._read_data(data)
        response = self._make_request(self._base_url, requests.post, json=payload)
        return response.json()["id"]

    def delete_program(self, program_id):
        url = self._base_url + f"/{program_id}"
        self._make_request(url, requests.delete)

    def set_program_visibility(self, program_id: str, public: bool):
        if public:
            url = self._base_url + f"/{program_id}/public"
        else:
            url = self._base_url + f"/{program_id}/private"

        self._make_request(url, requests.put)

    def programs(self):
        response = self._make_request(self._base_url, requests.get)
        programs = []
        for prog in response.json()["programs"]:
            programs.append(SimpleNamespace(**prog))
        return programs

    def program(self, program_id):
        url = self._base_url + f"/{program_id}"
        response = self._make_request(url, requests.get)
        return SimpleNamespace(**response.json())

    def _read_metadata(self, metadata):
        with open(metadata, 'r') as file:
            upd_metadata = json.load(file)
        metadata_keys = ['name', 'max_execution_time', 'description', 'spec']
        return {key: val for key, val in upd_metadata.items() if key in metadata_keys}

    def _read_data(self, data):
        with open(data, "r") as file:
            data = file.read()
        return base64.b64encode(data.encode('utf-8')).decode('utf-8')

    def _get_url_token(self):
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "").lower() == "true":
            return CLOUD_RUNTIME_URL_STAGING, os.getenv("QE_TOKEN_STAGING")
        return CLOUD_RUNTIME_URL_PROD, os.getenv("QE_TOKEN")

    def _make_request(self, url, func, **kwargs):
        headers = self._header.copy()
        headers.update(kwargs.pop("headers", {}))
        response = func(url, headers=headers, **kwargs)
        if response.status_code != requests.codes.ok:
            LOG.error("Bad request: %s", response.text)
        response.raise_for_status()
        return response
