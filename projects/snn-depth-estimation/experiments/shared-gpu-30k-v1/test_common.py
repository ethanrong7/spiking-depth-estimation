"""Tiny CPU checks: actual FIFO ordering/flags, RNG isolation, complete restore."""
import copy,json,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from common import Prefetch,decode,reconcile,flags,phase_lock

class TinyDataset:
    def __getitem__(self,index):
        x=torch.arange(12).reshape(1,3,4).float()+index*20
        return x,x+1,x*.1,torch.ones(3,4,dtype=torch.bool),index

class Tests(unittest.TestCase):
    def test_prefetch_identical_and_rng_unchanged(self):
        dataset=TinyDataset();order=torch.randperm(17,generator=torch.Generator().manual_seed(1234)).tolist()
        random.seed(5);np.random.seed(7);torch.manual_seed(9)
        before=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
        for start in (0,119,246):
            stream=Prefetch(dataset,order,start,start+5)
            for step in range(start,start+5):
                pref=stream.get(step);sync=decode(dataset,order,step)
                self.assertEqual(pref['indices'],sync['indices']);self.assertEqual(pref['flags'],sync['flags'])
                self.assertLessEqual(len(stream.pending),2)
                for a,b in zip(pref['items'],sync['items']):
                    for x,y in zip(a[:4],b[:4]):self.assertTrue(torch.equal(x,y))
            stream.close()
        self.assertEqual(random.getstate(),before[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],before[1][1]));self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
    def test_restore_optimizer_scheduler_rng_exact(self):
        torch.manual_seed(1234);model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
        schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer,2e-4,total_steps=30000,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
        def step():
            optimizer.zero_grad();model(torch.rand(2,4)).square().mean().backward();optimizer.step();schedule.step()
        step();snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),rng=torch.get_rng_state()))
        step();expected=copy.deepcopy(model.state_dict());expectedopt=copy.deepcopy(optimizer.state_dict());expectedrng=torch.get_rng_state()
        model.load_state_dict(snapshot['model']);optimizer.load_state_dict(snapshot['optimizer']);schedule.load_state_dict(snapshot['scheduler']);torch.set_rng_state(snapshot['rng']);step()
        for k,v in expected.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        self.assertTrue(torch.equal(torch.get_rng_state(),expectedrng));self.assertEqual(schedule.last_epoch,2)
        for k,v in expectedopt['state'].items():
            for name,x in v.items():self.assertTrue(torch.equal(x,optimizer.state_dict()['state'][k][name]))
    def test_reconcile_preserves_uncommitted_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train.jsonl';p.write_text(''.join(json.dumps(dict(update=i))+'\n' for i in range(1,6)))
            original=p.read_bytes();reconcile(p,lambda r:r['update']<=3)
            self.assertEqual(len(p.read_text().splitlines()),3);self.assertEqual(next((p.parent/'recovery').iterdir()).read_bytes(),original)
    def test_hv_seed_position_identity(self):
        import hashlib
        for i in range(40):
            d=hashlib.sha256(f'recipe-hv-v1:1234:{i}'.encode()).digest();self.assertEqual(flags(i),(d[0]<128,d[1]<128))

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
