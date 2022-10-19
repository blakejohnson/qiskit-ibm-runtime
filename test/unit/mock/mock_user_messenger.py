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

"""Mock for UserMessenger class from qiskit-ibm-runtime."""

from typing import Any


class MockUserMessenger:
    def __init__(self):
        self._messages = []

    def publish(self, msg: Any):
        """Publish a message"""
        self._messages.append(msg)

    def get_msg(self, index: int):
        """Get message at index"""
        return self._messages[index]
