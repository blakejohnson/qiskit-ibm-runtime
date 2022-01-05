from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase
from .utils import find_program_id


class MethodCallLogger:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args):
        self.func(*args)
        self.call_count += 1


class TestHelloWorld(BaseTestCase):
    """Test hello_world."""

    @classmethod
    @get_provider_and_backend 
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        interim_result_callback = MethodCallLogger(self.simple_callback)
        self.interim_result_callback = interim_result_callback

    def test_hello_world(self):
        """Test hello_world."""
        runtime_inputs = {
            "iterations": 2
        }
        options = {"backend_name": self.backend_name}
        job = self.provider.runtime.run(program_id=find_program_id(self.provider.runtime, "hello-world"),
                                        options=options,
                                        inputs=runtime_inputs,
                                        callback=self.interim_result_callback
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        expected_result = "All done!"
        self.assertEqual(job.result(), expected_result)
        self.assertEqual(self.interim_result_callback.call_count, runtime_inputs["iterations"])
