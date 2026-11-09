"""Rank-local preparation using the documented compact CPU input pipeline."""
import time
import torch
from input_pipeline import PreparedPrefetch
from common import flags


def prepare_rank_batch(dataset, order, step, seed, rank, world, global_batch=8, micro_batch=None):
    assert world in (1,4) and 0 <= rank < world
    started=time.monotonic()
    assert global_batch%world==0
    indices=[order[(step*global_batch+j)%len(order)] for j in range(global_batch)]
    hv=[list(flags(step*global_batch+j,seed)) for j in range(global_batch)]
    width=global_batch//world
    items=[]
    for slot in range(rank*width,(rank+1)*width):
        item=list(dataset[indices[slot]])
        horizontal,vertical=hv[slot]
        if horizontal:
            item[:4]=[x.flip(-1) for x in item[:4]]
            item[2]=-item[2]
        if vertical:item[:4]=[x.flip(-2) for x in item[:4]]
        items.append(item)
    micro_size=micro_batch or (4 if world==1 else 2)
    assert width%micro_size==0
    microbatches=[]
    for offset in range(0,width,micro_size):
        tensors=tuple(torch.stack([item[j] for item in items[offset:offset+micro_size]]) for j in range(4))
        microbatches.append(dict(tensors=tensors,valid_count=int(tensors[3].sum())))
    seconds=time.monotonic()-started
    return dict(step=step,indices=indices,flags=hv,microbatches=microbatches,
        valid_count=sum(m['valid_count'] for m in microbatches),
        host_batch_bytes=sum(t.numel()*t.element_size() for m in microbatches for t in m['tensors']),
        decode_seconds=seconds,input_prepare_seconds=seconds)


class RankPreparedPrefetch(PreparedPrefetch):
    def __init__(self,dataset,order,start,stop,seed,rank,world,workers=2,depth=2,global_batch=8,micro_batch=None):
        self.rank,self.world=rank,world
        self.global_batch,self.micro_batch=global_batch,micro_batch
        super().__init__(dataset,order,start,stop,seed,workers,depth)

    def _fill(self):
        while len(self.pending)<self.depth and self.next<self.stop:
            step=self.next;self.next+=1
            self.pending.append((step,self.pool.submit(prepare_rank_batch,
                self.dataset,self.order,step,self.seed,self.rank,self.world,self.global_batch,self.micro_batch)))
