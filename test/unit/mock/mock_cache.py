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

"""Mock for Cache class from ntc-provider."""

import json

from typing import Any

from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder


class MockCache:
    def __init__(self):
        self._cache = {}

    def set(self, key: str, value: Any):
        self._cache.update({key: json.dumps(value, cls=RuntimeEncoder)})

    def get(self, key: str) -> Any:
        return json.loads(self._cache.get(key), cls=RuntimeDecoder)
