"""Logical inference arithmetic observations, separate from measured GPU energy.

This dispatch observer returns each original operator's result unchanged. Dense
MACs use conventional padded convolution/matrix shape counts. A separate
hypothetical event-driven count exploits numerically binary activations; it is
not the work performed by the dense CUDA implementation. Unknown operators and
logical tensor bytes remain explicit. No CUDA work occurs at import time.
"""
from collections import Counter, defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensors(value):
    if isinstance(value,torch.Tensor):
        yield value
    elif isinstance(value,dict):
        for member in value.values():
            yield from tensors(member)
    elif isinstance(value,(tuple,list)):
        for member in value:
            yield from tensors(member)


def binary(value):
    return bool(((value==0)|(value==1)).all())


def fanout_axis(length,output,kernel,stride,padding,dilation,device,transposed=False):
    """Number of valid output receptive fields using each input coordinate."""
    position=torch.arange(length,device=device)
    counts=torch.zeros(length,dtype=torch.int64,device=device)
    for tap in range(kernel):
        if transposed:
            location=position*stride-padding+tap*dilation
            valid=(location>=0)&(location<output)
        else:
            location=position+padding-tap*dilation
            valid=(location>=0)&(location%stride==0)&(location//stride<output)
        counts+=valid.to(torch.int64)
    return counts


def convolution_counts(args,output):
    x,weight,bias,stride,padding,dilation,transposed,_,groups=args[:9]
    if x.ndim!=4 or weight.ndim!=4:
        return None
    kh,kw=weight.shape[-2:]
    dense=(x.numel()*(output.shape[1]//groups)*kh*kw if transposed
           else output.numel()*(x.shape[1]//groups)*kh*kw)
    result=Counter(dense_mac=dense,analog_mac=dense,explicit_bias_add=output.numel() if bias is not None else 0)
    if binary(x):
        h=fanout_axis(x.shape[-2],output.shape[-2],kh,stride[0],padding[0],dilation[0],x.device,transposed)
        w=fanout_axis(x.shape[-1],output.shape[-1],kw,stride[1],padding[1],dilation[1],x.device,transposed)
        additions=int((x.ne(0).to(torch.int64)*h[:,None]*w[None,:]).sum())*(output.shape[1]//groups)
        result.update(binary_input_convolution_mac=dense,potential_event_ac=additions)
        result['analog_mac']=0
    return result


def matrix_counts(a,b,output):
    if a.ndim<2 or b.ndim<2 or a.shape[-1]!=b.shape[-2]:
        return None
    dense=output.numel()*a.shape[-1]
    result=Counter(dense_mac=dense,analog_mac=dense)
    left,right=binary(a),binary(b)
    if left or right:
        batch=torch.broadcast_shapes(a.shape[:-2],b.shape[:-2])
        a=a.expand(*batch,*a.shape[-2:]);b=b.expand(*batch,*b.shape[-2:])
        if left and right:
            # Exact count of successful binary products without materializing
            # a query×key×channel tensor. All dense pair tests remain counted.
            count=int((a.to(torch.int64).sum(-2)*b.to(torch.int64).sum(-1)).sum())
            result['binary_product_tests']=dense
        elif left:
            count=int(a.ne(0).sum())*b.shape[-1]
        else:
            count=int(b.ne(0).sum())*a.shape[-2]
        result['potential_event_ac']=count
        result['analog_mac']=0
    return result


def operator_counts(func,args,kwargs,output):
    """Price one observed boundary; never execute/decompose the model op again."""
    name=str(func).split('.')[1]
    if name in ('convolution','_convolution','conv2d','conv_transpose2d'):
        values={field.name:field.default_value for field in func._schema.arguments}
        values.update({field.name:value for field,value in zip(func._schema.arguments,args)})
        values.update(kwargs)
        transposed=values.get('transposed',name=='conv_transpose2d')
        if isinstance(values['padding'],str):
            return None  # Asymmetric string padding has no audited fanout rule.
        normalized=(values['input'],values['weight'],values['bias'],values['stride'],values['padding'],
                    values['dilation'],transposed,values.get('output_padding',[0,0]),values['groups'])
        return convolution_counts(normalized,output)
    if name in ('mm','bmm','matmul'):
        return matrix_counts(args[0],args[1],output)
    if name=='einsum':
        equation,operands=args[:2]
        if equation.replace(' ','')!='aijk,aijh->ajkh' or len(operands)!=2:
            return None
        a,b=operands
        if a.ndim!=4 or b.ndim!=4 or a.shape[:3]!=b.shape[:3]:
            return None
        # Exact pinned stereo correlation: [B,D,H,Wl]×[B,D,H,Wr].
        return matrix_counts(a.permute(0,2,3,1),b.permute(0,2,1,3),output)
    if name=='addmm':
        count=matrix_counts(args[1],args[2],output)
        if count is not None:
            count['explicit_bias_add']+=output.numel() if kwargs.get('beta',1)!=0 else 0
            count['element_mul']+=output.numel() if kwargs.get('alpha',1)!=1 else 0
            count['element_mul']+=output.numel() if kwargs.get('beta',1) not in (0,1) else 0
        return count
    return None


class ArithmeticCounter(TorchDispatchMode):
    """Observe one inference call; instrumentation is excluded from latency."""
    metadata_ops={'view','_unsafe_view','reshape','permute','transpose','t','unsqueeze','squeeze',
                  'expand','as_strided','slice','select','detach','detach_','alias','lift_fresh','resolve_conj','resolve_neg',
                  'flatten','split','split_with_sizes','meshgrid','expand_as'}
    element_ops={'add','sub','rsub','mul','div','true_divide','floor_divide','remainder','pow','neg','abs',
                 'sqrt','rsqrt','exp','log','log1p','sigmoid','tanh','gelu','relu','relu_','clamp','clamp_min','clamp_max',
                 'eq','ne','gt','ge','lt','le','bitwise_and','bitwise_or','bitwise_not','logical_and','logical_or',
                 'logical_not','where','isfinite','isinf','isnan','maximum','minimum','round','floor','ceil','__and__'}
    memory_ops={'clone','copy_','_to_copy','cat','stack','constant_pad_nd','zeros','zeros_like','ones','ones_like',
                'empty','empty_like','empty_strided','new_zeros','new_ones','arange','full','full_like','eye',
                'zero_','fill_','masked_fill','masked_fill_','index','index_select','gather','scatter','repeat',
                'repeat_interleave','unfold','im2col','col2im','contiguous','pad','to','linspace'}
    composite_ops={'native_group_norm','native_batch_norm','_native_batch_norm_legit','_native_batch_norm_legit_no_training',
                   'native_layer_norm','_softmax','_log_softmax','softmax','upsample_bilinear2d','upsample_nearest2d',
                   'grid_sampler_2d','grid_sampler','group_norm','batch_norm','instance_norm',
                   'avg_pool2d','max_pool2d','max_pool2d_with_indices','adaptive_avg_pool2d','linalg_vector_norm'}

    def __init__(self,model=None):
        super().__init__()
        self.totals=Counter();self.operators=defaultdict(Counter);self.components=defaultdict(Counter)
        self.unknown=Counter();self.stack=[];self.handles=[];self.signatures={}
        if model is not None:
            def finish(module,args,output):
                self.stack.pop()
            for name,module in model.named_modules():
                label=name or '<model>'
                self.handles.append(module.register_forward_pre_hook(lambda m,a,n=label:self.stack.append(n)))
                self.handles.append(module.register_forward_hook(finish,always_call=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[];self.stack=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        output=func(*args,**kwargs)
        # PyTorch removes this mode while this handler runs. These observer
        # reductions do not recursively count themselves or change output.
        opname=str(func);name=opname.split('.')[1]
        inputs=list(tensors((args,kwargs)));outputs=list(tensors(output))
        if opname not in self.signatures:
            self.signatures[opname]={'schema':str(func._schema),
                'input_tensors':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in inputs],
                'output_tensors':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in outputs]}
        record=Counter(calls=1,logical_input_bytes=sum(t.numel()*t.element_size() for t in inputs),
                       logical_output_bytes=sum(t.numel()*t.element_size() for t in outputs))
        count=operator_counts(func,args,kwargs,output) if isinstance(output,torch.Tensor) else None
        if count is not None:
            pass
        elif name in self.element_ops:
            count=Counter({'element_'+name:sum(t.numel() for t in outputs)})
        elif name in ('sum','mean','amax','amin','all','any') and inputs and outputs:
            count=Counter({'reduction_'+name+'_input_elements':inputs[0].numel(),
                           'reduction_'+name+'_output_elements':outputs[0].numel()})
        elif name in self.composite_ops and outputs:
            count=Counter({name+'_output_elements':outputs[0].numel()})
        elif name in self.metadata_ops:
            count=Counter(metadata_only_calls=1)
        elif name in self.memory_ops:
            count=Counter(memory_or_reindex_calls=1)
        if count is None:
            self.unknown[opname]+=1
        else:
            record.update(count)
        self.totals.update(record);self.operators[opname].update(record)
        self.components[self.stack[-1] if self.stack else '<unscoped>'].update(record)
        return output

    def report(self):
        return {'schema':'stereo-study-logical-arithmetic-v1','torch':torch.__version__,
            'totals':dict(self.totals),'operators':{k:dict(v) for k,v in sorted(self.operators.items())},
            'components':{k:dict(v) for k,v in sorted(self.components.items())},'unclassified_operators':dict(self.unknown),
            'first_observed_signatures':self.signatures,
            'boundary_accounting':'One callback boundary priced once; its func executes with this mode removed, so high-level internal decompositions are not additionally counted',
            'einsum_support':'Only pinned aijk,aijh->ajkh row correlation; other equations remain unclassified',
            'scope':'Observed eager ATen calls, conventional dense MACs, and separately hypothetical numerically-binary event arithmetic',
            'dense_gpu_work':'Dense CUDA kernels do not become sparse because operands are binary',
            'event_assumptions':'Binary convolution skips zero input events; binary matrix pairs retain all dense binary-product tests plus successful accumulations; routing/weight access required',
            'logical_bytes':'Logical operand/result volume includes aliases and reuse; not measured DRAM/SRAM/cache traffic or energy',
            'composites':'Normalization, nonlinear, softmax, pooling, interpolation and state arithmetic remain separate operation vectors; no invented per-element pJ',
            'incomplete_costs':'Unclassified calls and foreign/non-ATen kernels are not assigned zero cost; no total hardware/neuromorphic energy claim',
            'not_timing':'Observer synchronizations/reductions are diagnostics and must never enter benchmark timing'}
