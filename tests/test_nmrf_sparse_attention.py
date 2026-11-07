import unittest

import torch

from probe_nmrf_sparse_attention import packed_dot, partition, selected_attention
from probe_nmrf_bsa import attention
from nmrf.models.NMP import WindowAttention


class SparseAttentionTests(unittest.TestCase):
    def test_event_messages_match_quantized_ann_and_row_budget(self):
        from probe_nmrf_sparse_attention import dense_probabilities,event_indices,event_message,quantized_event_message
        torch.manual_seed(23)
        module=WindowAttention(128,(2,2),1,4)
        module.relative_position_enc_table.data.normal_(0,.1)
        qkv=torch.randn(1,4,4,2,384)
        mask=torch.zeros(4,8,8)
        mask[:,:,-1]=-float('inf')
        probs=dense_probabilities(module,qkv,mask)
        torch.testing.assert_close(probs,attention(module,qkv,mask)[0],atol=1e-7,rtol=1e-6)
        idx=event_indices(probs,3)
        from probe_nmrf_sparse_attention import route
        routed,_=route(module,qkv,mask,None,'events',3/8)
        self.assertTrue(torch.equal(idx,routed))
        self.assertEqual(idx.shape[-1],3)
        self.assertTrue(torch.all(idx<7))
        torch.testing.assert_close(event_message(module,qkv,mask,idx),quantized_event_message(module,qkv,mask,idx),atol=1e-6,rtol=1e-5)

    def test_signed_packed_dot_matches_integer_dot(self):
        torch.manual_seed(21)
        for bits in (3,4):
            q=torch.randint(-2**(bits-1),2**(bits-1),(2,4,9,32))
            k=torch.randint(-2**(bits-1),2**(bits-1),(2,4,7,32))
            self.assertTrue(torch.equal(packed_dot(q,k,bits),q@k.transpose(-2,-1)))

    def test_selected_gathers_equal_dense_masked_attention(self):
        torch.manual_seed(22)
        for shift in (0,1):
            module=WindowAttention(128,(2,2),shift,4)
            module.relative_position_enc_table.data.normal_(0,.1)
            qkv=torch.randn(1,4,4,2,384)
            mask=torch.zeros(4,8,8)
            q,_,_,_=partition(module,qkv,mask)
            indices=torch.rand(*q.shape[:-1],q.shape[-2]).argsort(-1)[...,:3]
            # Dense comparator uses the exact same edge subset for each head.
            # Run each head's selected mask through the full dense equations.
            from einops import repeat
            qq,kk,vv,_=partition(module,qkv,mask)
            rpe=module.relative_position_enc_table[module.relative_position_index.reshape(-1)].view(4,4,4,-1)
            qe,ke,ve=repeat(rpe,'i j h c -> (i a) (j b) h c',a=2,b=2).chunk(3,-1)
            qq=qq*module.scale
            logits=qq@kk.transpose(-2,-1)+torch.einsum('bhic,ijhc->bhij',qq,ke)+torch.einsum('bhjc,ijhc->bhij',kk,qe*module.scale)
            keep=torch.zeros_like(logits,dtype=torch.bool).scatter_(-1,indices,True)
            probs=logits.masked_fill(~keep,-float('inf')).softmax(-1)
            expected=probs@vv+torch.einsum('bhij,ijhc->bhic',probs,ve)
            from probe_nmrf_sparse_attention import unpartition
            torch.testing.assert_close(selected_attention(module,qkv,mask,indices),unpartition(module,expected,qkv.shape),atol=1e-6,rtol=1e-5)
            allidx=torch.arange(8).expand(*q.shape[:-1],8)
            torch.testing.assert_close(selected_attention(module,qkv,mask,allidx),attention(module,qkv,mask)[1],atol=1e-6,rtol=1e-5)


if __name__=='__main__': unittest.main()
