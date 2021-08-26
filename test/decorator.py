from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

import os


def get_provider_and_backend(func):
    """Decorator to setup test with provider and backend.
    
    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    def _wrapper(*args, **kwargs):
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True":
            os.environ["QE_TOKEN"] = os.getenv("QE_TOKEN_STAGING", "")
            os.environ["QE_URL"] = os.getenv("QE_URL_STAGING", "")
        _enable_account(os.getenv("QE_TOKEN", ""), os.getenv("QE_URL", ""))

        backend = _get_backend()
        kwargs.update({'backend_name': backend.name(), "provider": backend.provider()})
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


def _get_backend():
    """Get the specified backend."""

    backend_name = os.getenv("QISKIT_IBM_DEVICE_STAGING", None) \
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
        else os.getenv("QISKIT_IBM_DEVICE", None)

    _backend = None
    provider = _get_custom_provider() or IBMQ.providers()[0]

    if backend_name:
        # Put desired provider as the first in the list.
        providers = [provider] + IBMQ.providers()
        for provider in providers:
            backends = provider.backends(name=backend_name)
            if backends:
                _backend = backends[0]
                break
    else:
        _backend = least_busy(provider.backends(min_num_qubits=5,
                                                operational=True,
                                                simulator=False,
                                                input_allowed='runtime'))

    if not _backend:
        raise Exception('Unable to find a suitable backend.')

    return _backend


def _get_custom_provider():
    """Find the provider for the specific hub/group/project, if any.

    Returns:
        Custom provider or ``None`` if default is to be used.
    """
    hgp = os.getenv("QISKIT_IBM_HGP_STAGING", None) \
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
        else os.getenv("QISKIT_IBM_HGP", None)
    if hgp:
        hgp = hgp.split("/")
        return IBMQ.get_provider(hub=hgp[0], group=hgp[1], project=hgp[2])

    return None  # No custom provider.
