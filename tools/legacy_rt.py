import os
import logging

import requests

LEGACY_RUNTIME_URL_PROD = "https://runtime-us-east.quantum-computing.ibm.com/"
LEGACY_RUNTIME_URL_STAGING = "https://runtime-us-east-dev.quantum-computing.ibm.com"
LOG = logging.getLogger(__name__)


class LegacyRuntimeClient:

    def __init__(self):
        url, token = self._get_url_token()
        self._base_url = url + "/programs"
        self._access_token = token
        self._header = {'Content-Type': 'application/json',
                        'Authorization': f"Bearer {self._access_token}"}

    def update_program_id(self, old_id, new_id):
        url = self._base_url + f"/{old_id}"
        self._make_request(url, requests.patch, json={"id": new_id})

    def _get_url_token(self):
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "").lower() == "true":
            return LEGACY_RUNTIME_URL_STAGING, os.getenv("QE_TOKEN_STAGING")
        return LEGACY_RUNTIME_URL_PROD, os.getenv("QE_TOKEN")

    def _make_request(self, url, func, **kwargs):
        headers = self._header.copy()
        headers.update(kwargs.pop("headers", {}))
        response = func(url, headers=headers, **kwargs)
        if response.status_code != requests.codes.ok:
            LOG.error("Bad request: %s", response.text)
        response.raise_for_status()
        return response
