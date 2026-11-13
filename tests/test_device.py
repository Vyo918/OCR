"""Device selection logic. Must run without torch installed."""
import unittest

from docsearch.device import requested_from_env, select_device


class SelectDevice(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(select_device("auto", True, True), "cuda")
        self.assertEqual(select_device("auto", True, False), "cuda")
        self.assertEqual(select_device("auto", False, True), "mps")
        self.assertEqual(select_device("auto", False, False), "cpu")

    def test_empty_request_means_auto(self):
        self.assertEqual(select_device("", False, True), "mps")
        self.assertEqual(select_device("  ", False, False), "cpu")

    def test_explicit_available_devices(self):
        self.assertEqual(select_device("cuda", True, False), "cuda")
        self.assertEqual(select_device("mps", False, True), "mps")
        self.assertEqual(select_device("cpu", True, True), "cpu")

    def test_explicit_unavailable_raises(self):
        with self.assertRaises(RuntimeError):
            select_device("cuda", False, True)
        with self.assertRaises(RuntimeError):
            select_device("mps", True, False)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError) as cm:
            select_device("tpu", True, True)
        self.assertIn("cpu", str(cm.exception))

    def test_case_insensitive(self):
        self.assertEqual(select_device("MPS", False, True), "mps")
        self.assertEqual(select_device(" CPU ", True, True), "cpu")
        self.assertEqual(select_device("Auto", False, True), "mps")


class FromEnv(unittest.TestCase):
    def test_unset_or_empty_is_auto(self):
        self.assertEqual(requested_from_env({}), "auto")
        self.assertEqual(requested_from_env({"DOCSEARCH_DEVICE": ""}), "auto")

    def test_value_is_normalised(self):
        self.assertEqual(requested_from_env({"DOCSEARCH_DEVICE": " MPS "}), "mps")


if __name__ == "__main__":
    unittest.main()
