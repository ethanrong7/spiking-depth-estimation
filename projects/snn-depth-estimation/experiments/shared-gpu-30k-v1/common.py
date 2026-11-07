"""Bounded CPU-only FIFO, deterministic decoding, atomic records and recovery."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import contextmanager
import fcntl,hashlib,json,time
from pathlib import Path


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def flags(position,seed=1234):
    h=hashlib.sha256(f'recipe-hv-v1:{seed}:{position}'.encode()).digest();return h[0]<128,h[1]<128

def decode(dataset,order,step):
    start=time.monotonic();indices=[order[(step*8+j)%len(order)] for j in range(8)];items=[];hv=[]
    for j,index in enumerate(indices):
        item=list(dataset[index]);h,v=flags(step*8+j)
        if h:item[:4]=[x.flip(-1) for x in item[:4]];item[2]=-item[2]
        if v:item[:4]=[x.flip(-2) for x in item[:4]]
        hv.append([h,v]);items.append(item)
    return dict(step=step,indices=indices,flags=hv,items=items,decode_seconds=time.monotonic()-start)

class Prefetch:
    """Exactly two outstanding decode futures, one CPU thread, ordered delivery.

    No RNG or CUDA calls. Pending futures are discarded on restore; absolute
    step addresses preserve every sample/flip without mutable loader state.
    """
    def __init__(self,dataset,order,start,stop):
        self.dataset,self.order,self.stop=dataset,tuple(order),stop;self.next=start
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bounded-cpu-decode');self.pending=deque()
        self._fill()
    def _fill(self):
        while len(self.pending)<2 and self.next<self.stop:
            step=self.next;self.next+=1;self.pending.append((step,self.pool.submit(decode,self.dataset,self.order,step)))
    def get(self,step):
        index,future=self.pending.popleft()
        if index!=step:raise RuntimeError('prefetch order mismatch')
        result=future.result();self._fill();return result
    def close(self):
        for _,future in self.pending:future.cancel()
        self.pool.shutdown(wait=True)


def reconcile(path,keep):
    """Keep only records committed by a complete checkpoint; preserve old bytes."""
    path=Path(path)
    if not path.exists():return
    original=path.read_bytes();rows=[json.loads(s) for s in original.decode().splitlines() if s.strip()]
    selected=[r for r in rows if keep(r)]
    if len(selected)!=len(rows):
        archive=path.parent/'recovery';archive.mkdir(exist_ok=True)
        (archive/(path.name+'.'+str(time.time_ns()))).write_bytes(original)
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in selected))

@contextmanager
def phase_lock(root,arm,validation=False,empty_cache=lambda:None,record_events=True):
    """Train steps share the lock; validation holds it exclusively.

    Pending validation blocks new train steps; waiting peers release cached
    allocations at their safe boundary. Model/Adam tensors remain resident.
    """
    root=Path(root);request=root/('validation-'+arm+'.request')
    if validation:request.touch()
    else:
        while list(root.glob('validation-*.request')):
            empty_cache();time.sleep(.05)
    with (root/'gpu-phase.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if validation else fcntl.LOCK_SH)
        if record_events:append(root/'gpu-phases.jsonl',dict(arm=arm,validation=validation,event='start',monotonic_ns=time.monotonic_ns()))
        try:yield
        finally:
            if record_events:append(root/'gpu-phases.jsonl',dict(arm=arm,validation=validation,event='end',monotonic_ns=time.monotonic_ns()))
            empty_cache();fcntl.flock(lock,fcntl.LOCK_UN)
            if validation:request.unlink(missing_ok=True)
