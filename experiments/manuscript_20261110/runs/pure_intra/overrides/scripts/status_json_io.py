"""Atomic JSON metadata I/O with bounded Windows sharing retries.

No training/checkpoint serialization is involved. Persistent write failures
remain errors and leave the unique temporary file available for diagnosis.
"""
import json
import os
from pathlib import Path
import tempfile
import time


def read(path):
    # Close the handle before decoding/parsing, reducing the Windows lock window.
    # Sharing DELETE alone does not make Python's MoveFileEx replacement safe.
    deadline = time.monotonic() + 3.0
    while True:
        try:
            payload = Path(path).read_bytes()
            return json.loads(payload.decode('utf-8-sig'))
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def write(path, value, *, retry_seconds=10.0):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
    deadline = time.monotonic() + retry_seconds
    delay = 0.02
    while True:
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32, 33)
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                if hasattr(exc, 'add_note'):
                    exc.add_note('Uncommitted JSON preserved at ' + temporary)
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.25)
