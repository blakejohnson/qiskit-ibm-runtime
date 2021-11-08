"""Handle git operations."""

import logging
import os
import subprocess

LOG = logging.getLogger(__name__)


def find_changes():
    commit_range = os.getenv("TRAVIS_COMMIT_RANGE")
    LOG.debug("Commit range is %s", commit_range)
    try:
        res = subprocess.run(['git', 'diff', '--name-status', f"{commit_range}"],
                             capture_output=True, check=True)
        LOG.debug("Files modified: %s", res.stdout)
    except subprocess.CalledProcessError as e:
        LOG.exception(
            'Failed to find changes\nstdout:\n%s\nstderr:\n%s\n'
            % (e.stdout, e.stderr))
        return None
    return res.stdout.decode()
