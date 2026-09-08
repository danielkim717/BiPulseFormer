"""Send a progress-report request to the existing Codex thread every 30 minutes.

The experiment and its frozen source are read-only to this helper. A stop file
named reporting.stop in the benchmark directory stops future report requests.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--thread', required=True)
    parser.add_argument('--codex', required=True)
    parser.add_argument('--interval', type=int, default=1800)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error('Interval must be at least 60 seconds')
    root = args.run.resolve()
    if not (root / 'status.json').is_file():
        raise FileNotFoundError(root / 'status.json')
    next_report = time.time() + args.interval
    state = {'pid': os.getpid(), 'thread': args.thread, 'interval_seconds': args.interval,
             'started_at': datetime.now().astimezone().isoformat(), 'status': 'running',
             'reports_queued': 0}
    while not (root / 'reporting.stop').exists():
        now = time.time()
        terminal = False
        try:
            status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
            terminal = status.get('status') in ('complete', 'failed')
        except (OSError, ValueError) as error:
            state['read_error'] = str(error)
        if now >= next_report or terminal:
            prompt = (
                '[사용자 요청에 따른 자동 진행 보고] '
                f'실험 폴더 {root}의 status.json, 실행 중 작업의 progress.json/log.txt, '
                '완료된 summary.json 및 comparison.md를 읽고 현재 진행 상황을 한국어로 간결하게 보고하세요. '
                '모델/방향/seed, epoch와 batch, 완료 실험 수, validation 결과, '
                '최종 test가 있다면 baseline 비교, 오류 또는 진행 정체 여부를 포함하세요. '
                '최종 test가 없으면 아직 없다고 명시하세요. 이번 요청은 읽기 전용 보고이며 '
                '학습 코드·설정·프로세스를 변경하거나 새 모니터를 시작하지 마세요. '
                '전체 실험이 완료되거나 실패하면 이 보고 모니터는 자동으로 종료됩니다.'
            )
            try:
                result = subprocess.run([args.codex, 'queue', '--thread', args.thread,
                                         '--message', prompt], capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=60)
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout).strip())
                state['reports_queued'] += 1
                state['last_queued_at'] = datetime.now().astimezone().isoformat()
                state.pop('delivery_error', None)
                next_report = time.time() + args.interval
                if terminal:
                    state['status'] = 'finished'
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                state['delivery_error'] = str(error)
                next_report = time.time() + 60
        state['heartbeat_at'] = datetime.now().astimezone().isoformat()
        state['next_report_at'] = datetime.fromtimestamp(next_report).astimezone().isoformat()
        save(root / 'reporting_status.json', state)
        if state['status'] == 'finished':
            return
        time.sleep(min(60, max(1, next_report - time.time())))
    state['status'] = 'stopped'
    save(root / 'reporting_status.json', state)


if __name__ == '__main__':
    main()
