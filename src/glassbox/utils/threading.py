import threading
import queue
import time
from typing import Callable, Dict, Any

class TrainingThread(threading.Thread):
    def __init__(self, train_func: Callable, training_queue: queue.Queue, params: Dict[str, Any]):
        super().__init__()
        self.train_func = train_func
        self.queue = training_queue
        self.params = params
        self.daemon = True # Allow app to exit even if thread is running
        self._stop_event = threading.Event()

    def run(self):
        try:
            self.train_func(self.queue, self.params, self._stop_event)
        except Exception as e:
            self.queue.put({"status": "error", "message": str(e)})
        finally:
            self.queue.put({"status": "finished"})

    def stop(self):
        self._stop_event.set()

def init_training_queue():
    return queue.Queue()
