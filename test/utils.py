"""Utility functions for testing."""


def find_program_id(service, program_name):
    """Find the actual program ID for the input program name.

    Args:
        service: Runtime service to use (e.g. `provider.runtime`)
        program_name: Name of the program.

    Returns:
        Program ID.
    """
    potential_id = None
    for program in service.programs():
        if program.name == program_name:
            return program.program_id
        elif program.name.startswith(program_name):
            potential_id = program.program_id
    return potential_id
