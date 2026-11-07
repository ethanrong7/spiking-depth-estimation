import unittest

import torch

from spiking_nmrf import LIF, SpikingStereoBackbone


class SpikingBackboneTests(unittest.TestCase):
    def test_matched_control_preserves_matcher_initialization_and_cpu_rng(self):
        from run_spiking_nmrf import pilot_definition
        ann_builder,ann_name,_=pilot_definition('ann')
        spike_builder,spike_name,_=pilot_definition('spiking')
        ann,_,_=ann_builder(1234)
        ann_rng=torch.get_rng_state().clone()
        spike,_,_=spike_builder(1234)
        self.assertTrue(torch.equal(ann_rng,torch.get_rng_state()))
        self.assertNotEqual(ann_name,spike_name)
        ann_state=ann.state_dict()
        compared=0
        for name,value in spike.state_dict().items():
            if name.startswith('parent.backbone.'):continue
            self.assertTrue(torch.equal(value,ann_state[name]),name)
            compared+=1
        self.assertGreater(compared,100)

    def test_temporal_integration_resets_between_calls_and_batch_items(self):
        neuron=LIF()
        x=torch.tensor([[1.2,0.],[1.2,0.],[1.2,0.],[1.2,0.],[1.2,0.]],requires_grad=True)
        expected=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,0.],[0.,0.]])
        result=neuron(x)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(neuron(x),expected))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[0,0]),0)

    def test_real_sensor_order_multiscale_interface_and_binary_states(self):
        torch.manual_seed(12)
        model=SpikingStereoBackbone().eval()
        bins=torch.arange(50).reshape(1,50,1,1).float()
        chunks=model.sensor_chunks(bins)
        self.assertEqual(chunks.shape,(5,1,10,1,1))
        self.assertTrue(torch.equal(chunks[3,0,:,0,0],torch.arange(30,40).float()))
        seen=[]
        handles=[m.register_forward_hook(lambda module,args,out:seen.append(bool(((out==0)|(out==1)).all())))
                 for m in model.modules() if isinstance(m,LIF)]
        x=torch.randint(0,2,(2,50,32,64)).float()*255
        with torch.no_grad():
            fine,coarse=model(x)
            fine2,coarse2=model(x)
        for h in handles:h.remove()
        self.assertEqual(fine.shape,(2,256,8,16))
        self.assertEqual(coarse.shape,(2,256,4,8))
        self.assertTrue(all(seen))
        self.assertTrue(torch.equal(fine,fine2))
        self.assertTrue(torch.equal(coarse,coarse2))


if __name__=='__main__':unittest.main()
