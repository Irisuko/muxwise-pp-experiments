from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p8a_runner as runner


def test_exact_v3_arguments_except_topology():
    name, cmd, env, port = runner.commands()[0]
    assert cmd[cmd.index('--tp')+1] == '2'
    assert cmd[cmd.index('--pp-size')+1] == '2'
    assert cmd[cmd.index('--pdmux-pp-max-inflight-prefill')+1] == '3'
    assert cmd[cmd.index('--pdmux-prefill-mlp-chunk-tokens')+1] == '4096'
    assert cmd[cmd.index('--mem-fraction-static')+1] == '0.71'
    assert cmd[cmd.index('--attention-backend')+1] == 'flashinfer'
    assert '--disable-custom-all-reduce' in cmd
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and port == 30000


def test_formal_markers_and_existing_results_block_replay(tmp_path):
    for directory in ('results', 'archive/superseded/results'):
        path = tmp_path / directory
        path.mkdir(parents=True)
        runner.assert_unused_formal(tmp_path)
        marker = path / f'{runner.LABEL}.formal-started.json'
        marker.write_text('{}')
        with pytest.raises(AssertionError, match='no retry'):
            runner.assert_unused_formal(tmp_path)
        marker.unlink()
        data = path / f'{runner.LABEL}_requests.jsonl'
        data.write_text('{}')
        with pytest.raises(AssertionError, match='no retry'):
            runner.assert_unused_formal(tmp_path)
        data.unlink()
