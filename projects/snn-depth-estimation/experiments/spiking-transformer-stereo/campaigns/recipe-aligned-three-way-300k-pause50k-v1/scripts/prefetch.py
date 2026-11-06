"""Bounded deterministic CPU-only loading using the unchanged frozen dataset/augmentation."""
from dataclasses import dataclass
import queue
import threading
import time


@dataclass
class Batch:
    global_index:int
    indices:list
    flags:list
    items:list
    loader_work_seconds:float


def decode_batch(dataset,recipe,order,seed,global_index):
    begun=time.monotonic()
    indices=recipe.batch_indices(order,global_index)
    flags=[recipe.paired_flags(seed,global_index*8+position) for position in range(8)]
    items=[recipe.augment(dataset[index],*flag) for index,flag in zip(indices,flags)]
    return Batch(global_index,indices,flags,items,time.monotonic()-begun)


class Prefetch:
    """One FIFO producer, depth2; never calls RNG, CUDA or changes torch thread settings."""
    def __init__(self,dataset,recipe,order,seed,start,stop,depth=2):
        if depth!=2:raise ValueError('registered queue depth is2')
        self.dataset=dataset;self.recipe=recipe;self.order=tuple(order);self.seed=seed
        self.start=start;self.stop=stop;self.queue=queue.Queue(maxsize=depth);self.closed=threading.Event()
        self.thread=threading.Thread(target=self._produce,name='deterministic-cpu-prefetch',daemon=True)
        self.thread.start()

    def _put(self,value):
        while not self.closed.is_set():
            try:self.queue.put(value,timeout=.1);return True
            except queue.Full:pass
        return False

    def _produce(self):
        try:
            for global_index in range(self.start,self.stop):
                if self.closed.is_set():return
                if not self._put(decode_batch(self.dataset,self.recipe,self.order,self.seed,global_index)):return
        except BaseException as error:self._put(error)

    def get(self,expected):
        while True:
            if self.closed.is_set():raise RuntimeError('prefetch closed')
            try:value=self.queue.get(timeout=1)
            except queue.Empty:
                if not self.thread.is_alive():raise RuntimeError('prefetch producer stopped before expected batch')
                continue
            if isinstance(value,BaseException):raise RuntimeError('CPU prefetch failed') from value
            if value.global_index!=expected:raise RuntimeError('prefetch FIFO/global update mismatch')
            return value

    def close(self):
        self.closed.set();self.thread.join()
        while True:
            try:self.queue.get_nowait()
            except queue.Empty:break

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
