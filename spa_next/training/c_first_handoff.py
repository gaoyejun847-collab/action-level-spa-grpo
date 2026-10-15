"""Preserve an in-flight quantization check and restart orchestration C first."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def alive(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ', 1)[1].split()[0] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root / 'status.json').read_text())
    parent, child = state['pid'], state['child_pid']
    assert state['stage'] == 'prm_quantization_gate'
    assert b'spa_next.training.run_v1' in Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'spa_next.prm' in Path(f'/proc/{child}/cmdline').read_bytes()
    os.kill(parent, signal.SIGSTOP)
    marker = dict(status='waiting_for_quantization', pid=os.getpid(),
                  supervisor_pid=parent, quantization_pid=child, order=['C', 'A', 'B'])
    path = root / 'c_first_handoff.json'
    path.write_text(json.dumps(marker, indent=2))
    while alive(child):
        time.sleep(15)
    if alive(parent):
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        while alive(parent):
            time.sleep(1)
    report = root / 'prm/frozen/quantization_report.json'
    if not report.exists() or not json.loads(report.read_text()).get('gate_passed'):
        marker['status'] = 'quantization_failed_no_policy_started'
        path.write_text(json.dumps(marker, indent=2))
        state.update(stage='prm_quantization_failed', child_pid=None,
                     error='Quantization check did not pass; see quantization report/log')
        (root / 'status.json').write_text(json.dumps(state, indent=2))
        return
    cmd = json.loads((root / 'resume-four-gpu.command.json').read_text())
    with (root / 'supervisor-c-first.log').open('a') as log:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    marker.update(status='c_first_pipeline_started', new_supervisor_pid=process.pid)
    path.write_text(json.dumps(marker, indent=2))


if __name__ == '__main__':
    main()
