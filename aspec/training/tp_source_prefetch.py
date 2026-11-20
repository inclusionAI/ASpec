# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded source GET prefetch; GPU collectives stay on the consumer thread."""

import queue
import threading


class TPSourcePrefetch:
    def __init__(self, read, release, *, depth):
        self.read, self.release = read, release
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.exhausted = False

    def _run(self):
        try:
            while not self.stop.is_set():
                item = self.read()
                while not self.stop.is_set():
                    try:
                        self.queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    self.release(item)
                    return
                if item[0] is None:
                    return
        except BaseException as error:
            self.error = error

    def next(self):
        if self.exhausted:
            return None, {}
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True, name="tp-source-get")
            self.thread.start()
        while not self.stop.is_set():
            if self.error is not None:
                raise self.error
            try:
                item = self.queue.get(timeout=0.1)
                self.exhausted = item[0] is None
                return item
            except queue.Empty:
                continue
        raise StopIteration

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("TP source GET did not stop; retain backing storage until the read fence")
        while True:
            try:
                self.release(self.queue.get_nowait())
            except queue.Empty:
                return
