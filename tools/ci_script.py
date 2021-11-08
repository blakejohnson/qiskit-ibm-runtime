import re
import logging
import os
import sys
import json

import yaml
from qiskit import IBMQ
from qiskit.providers.ibmq.runtime.exceptions import RuntimeProgramNotFound

from tools import git
from tools.cloud_rt import CloudRuntimeClient
from tools.legacy_rt import LegacyRuntimeClient


PROGRAM_DIR = "runtime"
PROGRAM_ID_FILE = "program_ids.txt"
PROGRAM_CONFIG_FILE = "program_config.yaml"
VALID_PROGRAM_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$"
LOG = logging.getLogger("tools.ci_script")


def find_changed_programs():
    added = set()
    modified = set()
    deleted = set()
    git_diff = git.find_changes().split("\n")
    del git_diff[-1]  # Delete blank entry
    program_dir = PROGRAM_DIR + "/"
    for line in git_diff:
        line_list = line.split()
        if len(line_list) == 2:
            status, name = line_list
        elif len(line_list) == 3:
            status, src, name = line_list
        else:
            continue
        if not name.startswith(program_dir):
            continue

        try:
            name = re.match(rf"{program_dir}(\w+)\..+", name).group(1)
        except AttributeError:
            LOG.error("Unable to parse file name %s", name)
            continue
        if status == "D":
            deleted.add(name)
        elif status == "M":
            modified.add(name)
        elif status.startswith("R"):
            added.add(name)
            try:
                src = re.match(rf"{program_dir}(\w+)\..+", src).group(1)
                deleted.add(src)
            except AttributeError:
                pass
        elif status.startswith("A") or status.startswith("C"):
            added.add(name)

    LOG.debug("Git diff, added=%s, modified=%s, deleted=%s", added, modified, deleted)
    return added, modified, deleted


def get_provider():
    if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "").lower() == "true":
        qe_token = os.getenv("QE_TOKEN_STAGING")
        qe_url = os.getenv("QE_URL_STAGING")
        LOG.debug("Using staging environment")
    else:
        qe_token = os.getenv("QE_TOKEN")
        qe_url = os.getenv("QE_URL")
        LOG.debug("Using production environment")

    try:
        IBMQ.disable_account()
    except:
        pass
    return IBMQ.enable_account(qe_token, url=qe_url)


def find_program_id(pgm_fn, runtime):
    pgm_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), PROGRAM_DIR)
    metadata_file = os.path.join(pgm_dir, f"{pgm_fn}.json")
    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    potential_id = None
    # TODO: Use filtering when supported
    for prog in runtime.programs():
        if prog.name == metadata["name"]:
            if prog.program_id == metadata["name"]:
                return prog.program_id
            else:
                potential_id = prog.program_id
    if potential_id:
        return potential_id
    raise ValueError("Unable to find program ID for %s using %s", pgm_fn, runtime)


def update_programs():
    added, modified, deleted = find_changed_programs()

    runtime = get_provider().runtime
    cloud_programs = get_cloud_programs()
    cloud_client = CloudRuntimeClient()

    program_ids = _batch_update(added, "upload", runtime)
    program_ids += _batch_update(modified, "update", runtime)
    _batch_delete(deleted, runtime)

    cloud_pids = _batch_update(added & cloud_programs, "upload", cloud_client)
    cloud_pids += _batch_update(modified & cloud_programs, "update", cloud_client)
    _batch_delete(deleted & cloud_programs, cloud_client)

    with open(PROGRAM_ID_FILE, "w") as file:
        json.dump({"legacy": program_ids, "cloud": cloud_pids}, file)


def _batch_update(programs, func, runtime):
    pgm_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), PROGRAM_DIR)
    program_ids = []
    for pgm in programs:
        data_file = os.path.join(pgm_dir, f"{pgm}.py")
        metadata_file = os.path.join(pgm_dir, f"{pgm}.json")
        if func == "upload":
            program_id = runtime.upload_program(data=data_file, metadata=metadata_file)
            LOG.debug("Uploaded new program %s using %s", program_id, runtime)
            program_ids.append(program_id)
        else:
            program_id = find_program_id(pgm, runtime)
            runtime.update_program(
                program_id=program_id, data=data_file, metadata=metadata_file)
            LOG.debug("Updated program %s using %s", program_id, runtime)

    return program_ids


def _batch_delete(programs, runtime):
    for pgm in programs:
        program_id = find_program_id(pgm, runtime)
        _delete_program(program_id, runtime)


def _delete_program(program_id, runtime):
    try:
        runtime.delete_program(program_id)
        LOG.debug("Deleted program %s using %s", program_id, runtime)
    except RuntimeProgramNotFound:
        LOG.warning("Unable to delete program % using %s. Program doesn't exist.",
                    program_id, runtime)


def make_public():
    with open(PROGRAM_ID_FILE, "r") as file:
        program_ids = json.load(file)

    runtime = get_provider().runtime
    cloud_client = CloudRuntimeClient()
    legacy_client = LegacyRuntimeClient()

    new_iqx_ids = _gather_new_ids(program_ids["legacy"], runtime)
    new_cloud_ids = _gather_new_ids(program_ids["cloud"], cloud_client)

    runtime.update_program_id = legacy_client.update_program_id
    _batch_publish(program_ids["legacy"], new_iqx_ids, runtime)
    _batch_publish(program_ids["cloud"], new_cloud_ids, cloud_client)


def _batch_publish(cur_ids, new_ids, runtime):
    for idx, pid in enumerate(cur_ids):
        runtime.set_program_visibility(program_id=pid, public=True)
        LOG.debug("Made program %s public using %s", pid, runtime)
        runtime.update_program_id(pid, new_ids[idx])
        LOG.debug("Updated ID for program %s to %s using %s", pid, new_ids[idx], runtime)


def _gather_new_ids(program_ids, runtime):
    new_ids = []
    for pid in program_ids:
        new_id = runtime.program(pid).name
        if not re.match(VALID_PROGRAM_ID_PATTERN, new_id):
            # Try a small fix
            new_pid = new_id.replace("_", "-")
            if not re.match(VALID_PROGRAM_ID_PATTERN, new_pid):
                raise ValueError(f"Name of program {pid} cannot be used as new program ID.")
        new_ids.append(new_id)
    return new_ids


def cleanup():
    if not os.path.exists(PROGRAM_ID_FILE):
        return

    with open(PROGRAM_ID_FILE, "r") as file:
        program_ids = json.load(file)
    if not program_ids:
        return

    runtime = get_provider().runtime
    cloud_client = CloudRuntimeClient()

    _batch_cleanup(program_ids["legacy"], runtime)
    _batch_cleanup(program_ids["cloud"], cloud_client)


def _batch_cleanup(program_ids, runtime):
    for pid in program_ids:
        try:
            pgm = runtime.program(pid)
            if not getattr(pgm, "is_public", False):
                _delete_program(pid, runtime)
        except:
            pass


def get_cloud_programs():
    pgm_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), PROGRAM_CONFIG_FILE)
    with open(pgm_config, "r") as file:
        config = yaml.safe_load(file)
    return set(config.get("cloud_runtime_programs", []))


def main(func):
    if func == "update":
        update_programs()
    elif func == "publish":
        make_public()
    elif func == "cleanup":
        cleanup()
    else:
        raise ValueError(f"Unknown function {func} requested.")


if __name__ == "__main__":
    main(sys.argv[1])
