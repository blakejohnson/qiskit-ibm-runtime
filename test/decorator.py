from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

import os
from unittest import SkipTest

def get_provider_and_backend(func):
    """Decorator to setup test with provider and backend.
    
    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    def _wrapper(*args, **kwargs):
        hgp = os.getenv("QISKIT_IBM_HGP_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_HGP", None)
        if not hgp:
            raise SkipTest("Requires ibm provider.")
        hgp = hgp.split("/")
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True":
            os.environ["QE_TOKEN"] = os.getenv("QE_TOKEN_STAGING", "")
            os.environ["QE_URL"] = os.getenv("QE_URL_STAGING", "")
        _enable_account(os.getenv("QE_TOKEN", ""), os.getenv("QE_URL", "")) 
        provider = IBMQ.get_provider(hub=hgp[0], group=hgp[1], project=hgp[2])
        backend_name = os.getenv("QISKIT_IBM_DEVICE_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_DEVICE", None)          
        if not backend_name:
            backend_name = least_busy(provider.backends(min_num_qubits=10,
                                      operational=True)).name()
        kwargs.update({'backend_name': backend_name, "provider": provider})
        return func(*args, **kwargs)
    return _wrapper

def _enable_account(qe_token: str, qe_url: str) -> None:
    """Enable the account if one is not already active.

    Args:
        qe_token: API token.
        qe_url: API URL.
    """
    active_account = IBMQ.active_account()
    if active_account:
        if active_account.get('token', '') == qe_token:
            return
        IBMQ.disable_account()
    IBMQ.enable_account(qe_token, url=qe_url)

