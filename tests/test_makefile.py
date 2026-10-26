"""Exercise Make recipes with an argument-recording stub, without installing or launching basal."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MakefileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.stub = directory / "fake python"
        self.stub.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c "
            "'import json, sys; print(json.dumps(sys.argv[1:]))' \"$@\"\n"
        )
        self.stub.chmod(0o755)
        self.input = directory / "input with spaces.jsonl"
        self.input.write_text('{}\n')

    def command(self, target, **variables):
        # Do not inherit command-line overrides from an enclosing `make test`.
        inherited = {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "PYTHON", "UV", "EXTRAS", "MODEL", "MODE",
                     "HOST", "PORT", "URL", "INPUT", "OUTPUT", "GGUF", "OLLAMA_MODEL", "ARGS"}
        return subprocess.run(
            ["make", "--no-print-directory", "-s", target, f"PYTHON={self.stub}", f"UV={self.stub}",
             *(f"{key}={value}" for key, value in variables.items())],
            cwd=ROOT, capture_output=True, text=True,
            env={key: value for key, value in os.environ.items() if key not in inherited},
        )

    def arguments(self, target, **variables):
        result = self.command(target, **variables)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_defaults_and_development_commands(self):
        self.assertIn("make serve", self.command("help").stdout)
        self.assertEqual(self.arguments("setup", EXTRAS="test,mlx"),
                         ["pip", "install", "--python", str(self.stub), "-e", ".[test,mlx]"])
        self.assertEqual(self.arguments("check"), ["tool", "run", "pre-commit==4.7.0", "run", "--all-files"])
        self.assertEqual(self.arguments("test"), ["-m", "pytest"])
        # MODEL empty: the CLIs pick their own default model; the server listens on this machine only
        self.assertEqual(self.arguments("serve"), ["-m", "basal.server", "--host", "127.0.0.1", "--port", "8000"])
        self.assertEqual(self.arguments("bench"), ["-m", "basal.bench"])

    def test_modes_and_quoted_values(self):
        modes = "eager fast fast-nocompile fast-exit fp8 nvfp4 vllm sglang mlx mlx-q8 mps gguf strix tpu ollama llamacpp"
        for mode in modes.split():  # basal.server.MODES (2.4.1)
            with self.subTest(mode=mode):
                args = self.arguments("serve", MODE=mode, MODEL="owner's model; literal", GGUF=self.input,
                                      OLLAMA_MODEL="local import", HOST="127.0.0.1", PORT="8001",
                                      ARGS='--name "Friendly server"')
                self.assertEqual(args, ["-m", "basal.server", "--model", "owner's model; literal",
                                       "--host", "127.0.0.1", "--port", "8001", "--mode", mode,
                                       "--gguf", str(self.input), "--ollama-model", "local import",
                                       "--name", "Friendly server"])
        for target, input_flag, output_flag in (("run", "--input", "--output"),
                                                ("bench", "--questions", "--out"),
                                                ("loadtest", "--questions", "--out")):
            with self.subTest(target=target):
                args = self.arguments(target, INPUT=self.input, OUTPUT="answers with spaces.jsonl", PORT="8001")
                self.assertEqual(args[args.index(input_flag) + 1], str(self.input))
                self.assertEqual(args[args.index(output_flag) + 1], "answers with spaces.jsonl")
                if target != "bench":
                    self.assertEqual(args[args.index("--url") + 1], "http://127.0.0.1:8001/v1/systemone")
        self.assertEqual(self.arguments("bench", MODE="mlx")[-2:], ["--modes", "mlx"])
        self.assertEqual(self.arguments("bench", ARGS="--modes mps mlx")[-3:], ["--modes", "mps", "mlx"])

    def test_missing_inputs_fail_before_launch(self):
        cases = [(target, variables) for target in ("serve", "bench") for variables in
                 ({"MODE": "gguf"}, {"MODE": "gguf", "GGUF": "missing.gguf"}, {"MODE": "ollama"})]
        cases += [("run", {}), ("run", {"INPUT": self.input})]
        cases += [(target, {"INPUT": "missing.jsonl", "OUTPUT": "out.jsonl"})
                  for target in ("run", "bench", "loadtest")]
        same = str(Path(self.input).parent / "." / Path(self.input).name)  # the input under another spelling
        cases += [(target, {"INPUT": self.input, "OUTPUT": output})
                  for target in ("run", "bench", "loadtest") for output in (str(self.input), same)]
        for target, variables in cases:
            with self.subTest(target=target, variables=variables):
                result = self.command(target, **variables)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_interpreter_path_with_spaces(self):
        environment = Path(self.temporary.name) / "venv with spaces"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
        result = subprocess.run(
            [str(environment / "bin/python"), "-B", str(Path(__file__).resolve()),
             "MakefileTest.test_modes_and_quoted_values"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
