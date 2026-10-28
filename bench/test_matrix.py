"""Command construction and formal replay lock tests; never start a GPU process."""
import argparse
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p7a_runner as runner


class MatrixTests(unittest.TestCase):
    def args(self, label):
        family = runner.MATRIX[label][0]
        return argparse.Namespace(label=label, mem=runner.DEFAULT_MEM[family],
                                  graph_bs=32, attention="flashinfer", no_graph=False)

    def test_eleven_four_gpu_configurations_and_exclusions(self):
        self.assertEqual(len(runner.MATRIX), 11)
        for label, (family, tp, pp, dp) in runner.MATRIX.items():
            self.assertEqual(tp * pp * dp, 4)
            jobs = runner.make_commands(self.args(label))
            workers = [j for j in jobs if j[0] != "router"]
            self.assertEqual(len(workers), dp)
            self.assertEqual(len(set(",".join(j[2]["CUDA_VISIBLE_DEVICES"] for j in workers).split(","))), 4)
            for _, cmd, env, _ in workers:
                self.assertEqual(cmd[cmd.index("--tp") + 1], str(tp))
                self.assertEqual(cmd[cmd.index("--pp-size") + 1], str(pp))
                self.assertIn("--enable-cache-report", cmd)
                if pp > 1:
                    self.assertNotIn("--enable-mixed-chunk", cmd)
                if family == "muxwise":
                    self.assertNotIn("--pdmux-prefill-mlp-chunk-tokens", cmd)
                if family == "pd":
                    self.assertEqual(env["MC_TCP_SLICE_SIZE"], "1048576")
                    self.assertEqual(env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"], "3600")
            if dp > 1:
                router = jobs[-1][1]
                self.assertEqual(router[router.index("--policy")+1], "random" if family == "pd" else "cache_aware")

    def test_muxpp_preserves_selected_pp4_flags(self):
        script = (runner.ROOT / "scripts/p5b_v3_serve.sh").read_text()
        expected = [runner.base.PYTHON] + shlex.split(script.split("exec python ", 1)[1].replace("\\\n", " "))
        for key, value in (("--pp-size", "2"), ("--port", "31000"), ("--pdmux-pp-max-inflight-prefill", "3")):
            expected[expected.index(key)+1] = value
        self.assertEqual(runner.make_commands(self.args("muxpp_pp2dp2"))[0][1], expected)

    def test_existing_formal_marker_prevents_client_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"results").mkdir()
            marker = root/"results/chunked_tp4.formal-started.json"
            marker.write_text('{"existing":true}')
            cluster = argparse.Namespace(label="chunked_tp4", path=root, meta={})
            with patch.object(runner, "ROOT", root), patch.object(runner.subprocess, "Popen") as spawn:
                with self.assertRaises(FileExistsError):
                    runner.replay(cluster, "formal")
                spawn.assert_not_called()
            self.assertEqual(marker.read_text(), '{"existing":true}')
            archive = root/"archive/superseded/results"
            archive.mkdir(parents=True)
            marker.rename(archive/marker.name)
            with patch.object(runner.base, "ROOT", root), patch.object(runner.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "Archived formal"):
                    runner.base.bench(cluster)
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
