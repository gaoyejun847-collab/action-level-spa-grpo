"""Detach the running PRM stage from later pipeline stages without restarting it."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root / 'status.json').read_text())
    parent, child = state['pid'], state['child_pid']
    assert state['stage'] == 'prm_training'
    assert b'spa_next.training.run_v1' in Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'spa_next.prm' in Path(f'/proc/{child}/cmdline').read_bytes()
    # Suspend only the supervisor, not torchrun or its workers. This makes it
    # impossible for the existing in-memory pipeline to launch a later stage.
    os.kill(parent, signal.SIGSTOP)
    control = dict(request='finish_prm_then_stop', monitor_pid=os.getpid(),
                   supervisor_pid=parent, training_pid=child,
                   requested_utc=datetime.now(timezone.utc).isoformat(),
                   status='armed', later_stages_disabled=True)
    (root / 'finish_prm_only.json').write_text(json.dumps(control, indent=2))
    print(json.dumps(control), flush=True)
    while alive(child):
        time.sleep(15)
    # Deliver the supervisor's existing termination handler before resuming it;
    # the completed child is safe, and execute() cannot advance to export.
    if alive(parent):
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        while alive(parent):
            time.sleep(1)
    report_path = root / 'prm/training/training_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    completed = report.get('updates') == 6090 and not report.get('smoke_only', True)
    control.update(status='finished' if completed else 'training_failed',
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   training_report=str(report_path), gate_passed=report.get('gate_passed'))
    (root / 'finish_prm_only.json').write_text(json.dumps(control, indent=2))
    state.update(stage='prm_finished_stopped_by_request' if completed else 'prm_failed_stopped',
                 child_pid=None, stop_after='prm_training',
                 finished_utc=control['finished_utc'], prm_gate_passed=report.get('gate_passed'))
    (root / 'status.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(control), flush=True)


if __name__ == '__main__':
    main()
