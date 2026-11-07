import unittest
import torch
from probe_nmrf_bsa import code, split_projection, modify, signed_unary


class BSAProbeTests(unittest.TestCase):
    def test_unary_encoder_event_budget_sign_and_rounding(self):
        x=torch.tensor([-9., -3.6, -2.5, -.49, 0., .49, .5, 1.5, 3.6, 9.])
        events=signed_unary(x,4)
        self.assertEqual(events.shape,(4,10))
        self.assertTrue(set(events.flatten().tolist()) <= {-1.,0.,1.})
        torch.testing.assert_close(events.sum(0),torch.tensor([-4.,-4.,-3.,0.,0.,0.,1.,2.,4.,4.]))
        self.assertTrue(torch.all(events.abs().sum(0)<=4))

    def test_richer_codes_match_equal_level_ann(self):
        torch.manual_seed(9)
        x=torch.randn(2,3,128,requires_grad=True)*4
        cal={'clip':torch.tensor([1.,2.,3.,4.])}
        for steps in (2,4,8):
            spikes,count=code(x,cal,f'unary{steps}')
            ann,integer=code(x,cal,f'quant{steps}')
            self.assertTrue(torch.equal(count,integer))
            self.assertTrue(torch.equal(spikes,ann))

    def test_signed_code_and_straight_through_gradient(self):
        x=torch.tensor([[-2.,-.2,0.,.2,2.]],requires_grad=True)
        cal={'clip':torch.tensor([2.]),'05':{'threshold':torch.tensor([.5]),'scale':torch.tensor([1.5])}}
        decoded,events=code(x,cal,'ternary_05',ste=True)
        torch.testing.assert_close(decoded,torch.tensor([[-1.5,0.,0.,0.,1.5]]))
        self.assertEqual(set(events.detach().flatten().tolist()),{-1.,0.,1.})
        decoded.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_content_coding_preserves_geometry_and_values(self):
        torch.manual_seed(8)
        module=torch.nn.Linear(159,384)
        x=torch.randn(1,2,2,4,159)
        output=module(x)
        content,geometry=split_projection(module,x,128)
        torch.testing.assert_close(content+geometry,output,atol=1e-6,rtol=1e-5)
        cal={key:{'clip':torch.ones(4),'05':{'threshold':torch.full((4,),1e6),'scale':torch.ones(4)}} for key in ('q','k')}
        result=modify(output,content,geometry,cal,'content_ternary_05')
        torch.testing.assert_close(result[...,:256],geometry[...,:256])
        self.assertTrue(torch.equal(result[...,256:],output[...,256:]))


if __name__=='__main__':unittest.main()
