import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PATH = Path(__file__).parents[1] / "python/llaisys/models/deepseek_v4_backends.py"
SPEC = importlib.util.spec_from_file_location("deepseek_v4_backend_contract", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = MODULE.OperatorRegistry()
        self.selection = {name: "baseline" for name in MODULE.CONTRACTS}
        for name, contract in MODULE.CONTRACTS.items():
            self.registry.register("baseline", name, lambda x: x, version="1", contract=contract)

    def test_requires_complete_selection(self):
        self.selection.pop("fp4_gemm")
        with self.assertRaises(MODULE.BackendSelectionError):
            self.registry.bind(self.selection)

    def test_rejects_unknown_operator(self):
        self.selection["made_up"] = "baseline"
        with self.assertRaises(MODULE.BackendSelectionError):
            self.registry.bind(self.selection)

    def test_no_implicit_fallback(self):
        self.selection["fp4_gemm"] = "unavailable"
        with self.assertRaises(MODULE.BackendSelectionError):
            self.registry.bind(self.selection)

    def test_rejects_duplicate_registration(self):
        with self.assertRaises(MODULE.BackendSelectionError):
            self.registry.register("baseline", "fp4_gemm", lambda: None,
                                   version="2", contract=MODULE.CONTRACTS["fp4_gemm"])

    def test_rejects_different_contract(self):
        with self.assertRaises(MODULE.BackendSelectionError):
            self.registry.register("w4a4", "fp4_gemm", lambda: None, version="1",
                                   contract=MODULE.OperatorContract(1, "W4A4", "BF16"))

    def test_per_operator_override(self):
        self.registry.register("custom", "fp4_gemm", lambda x: x + 1, version="2",
                               contract=MODULE.CONTRACTS["fp4_gemm"])
        self.selection["fp4_gemm"] = "custom"
        bound = self.registry.bind(self.selection)
        self.assertEqual(bound.function("fp4_gemm")(5), 6)
        self.assertEqual(bound.function("fp8_gemm")(5), 5)
        self.assertEqual(bound.report()["fp4_gemm"]["backend"], "custom")
        self.assertEqual(bound.report()["fp4_gemm"]["calls"], 1)

    def test_failure_propagates_and_is_counted(self):
        def fail(*args):
            raise RuntimeError("launch failed")
        self.registry.register("broken", "sparse_attn", fail, version="1",
                               contract=MODULE.CONTRACTS["sparse_attn"])
        self.selection["sparse_attn"] = "broken"
        bound = self.registry.bind(self.selection)
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            bound.function("sparse_attn")(0)
        self.assertEqual(bound.report()["sparse_attn"]["failures"], 1)

    def test_binding_is_a_snapshot(self):
        bound = self.registry.bind(self.selection)
        self.selection["fp4_gemm"] = "missing"
        report = bound.report()
        report["fp4_gemm"]["calls"] = 999
        self.assertEqual(bound.function("fp4_gemm")(7), 7)
        self.assertEqual(bound.report()["fp4_gemm"]["calls"], 1)

    def test_install_validates_before_mutation(self):
        bound = self.registry.bind(self.selection)
        marker = object()
        model = SimpleNamespace(fp4_gemm=marker)
        with self.assertRaises(MODULE.BackendSelectionError):
            bound.install_on(model)
        self.assertIs(model.fp4_gemm, marker)
        model = SimpleNamespace(**{name: marker for name in MODULE.CONTRACTS})
        bound.install_on(model)
        self.assertEqual(model.fp4_gemm(3), 3)


if __name__ == "__main__":
    unittest.main()
