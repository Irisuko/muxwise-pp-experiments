"""Selection must use full-denominator SLO and handle censored median ties."""
import unittest

from summarize import MAIN_SLO, select_best


class BestConfigurationTests(unittest.TestCase):
    def candidate(self, system, label, passed, p50, throughput=1):
        return ({"label": label, "matrix": {"system": system},
                 "slo": {MAIN_SLO: {"passed": passed, "total": 1551, "rate": passed/1551}},
                 "latencies_sec": {"ttft": {"p50": p50}},
                 "throughput": {"output_tokens_per_sec": throughput}}, [])

    def test_slo_before_latency_and_throughput_censored_tiebreak(self):
        runs = [
            self.candidate("Chunked", "more_slo", 101, 9),
            self.candidate("Chunked", "faster_but_less_slo", 100, 1, 9999),
            self.candidate("MuxWise", "censored_tie", 10, ">cutoff"),
            self.candidate("MuxWise", "finite_tie", 10, 500),
            self.candidate("PD", "slower_tie", 5, 4),
            self.candidate("PD", "faster_tie", 5, 3),
            self.candidate("MuxWise-PP", "only", 20, 6),
        ]
        self.assertEqual([s["label"] for s, _ in select_best(runs)],
                         ["more_slo", "finite_tie", "faster_tie", "only"])


if __name__ == "__main__":
    unittest.main()
