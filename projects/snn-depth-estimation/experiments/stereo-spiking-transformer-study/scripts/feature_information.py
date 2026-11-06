"""Descriptive, output-preserving temporal feature and matcher probes.

Real model execution belongs to registered evaluate_functionality Slurm runs.
This module never selects a checkpoint or changes accuracy/matching support.
Hooks detach only telemetry; each hook returns None and never changes state.
"""
from contextlib import AbstractContextManager
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F
from core.study_models import TemporalNeuron


def probe_contract():
    return dict(enabled=True,purpose='descriptive_only_not_checkpoint_selection',
        case_selection='existing_first_middle_last_Val_IDs_correct_right_only',
        expected_calls_per_view=50,spatial_variance_threshold=1e-12,
        padding='pad16_then_ceil_source_extent_per_stride;pure_padding_excluded',
        output_summaries='all20_correct_right_API_only;no_confidence_filter',
        legacy_baseline='unsupported_do_not_rerun')


def padded_shape(shape):
    h,w=map(int,shape)
    if min(h,w)<=0:
        raise ValueError('Positive original sensor H,W required')
    return ((h+15)//16*16,(w+15)//16*16)


def valid_extent(shape,stride):
    padded=padded_shape(shape)
    return tuple((int(value)+stride-1)//stride for value in shape),tuple(value//stride for value in padded)


def neuron_layout(name):
    match=re.match(r'^(stages|blocks)\.([0-3])\.',name)
    if not match:
        raise ValueError('Unregistered TemporalNeuron scale path: '+name)
    return dict(stride=2**(int(match.group(2))+1),
                role='global_channel_gate' if name.endswith('.gate') else 'spatial_feature')


class TemporalFeatureProbe(AbstractContextManager):
    """Collect tiny detached tensors on the forward device; serialize afterward.

    Time-mean within-spatial channel variance is distinct from temporal variance
    of spatial channel means. A uniform alternating0/1 spatial clock can have
    mean.5 and temporal variance.25 while carrying zero spatial variation.
    Spatially constant statistics describe activity, not architectural failure.
    Gates and any single-cell support have no assessable spatial variance.
    """
    def __init__(self,model,input_shape,*,expected_calls=50,require_cuda=True):
        self.model=model;self.input_shape=tuple(map(int,input_shape))
        self.padded_shape=padded_shape(self.input_shape)
        self.expected_calls=expected_calls;self.require_cuda=require_cuda
        self.nodes={name:(module,neuron_layout(name)) for name,module in model.named_modules() if isinstance(module,TemporalNeuron)}
        if not self.nodes:
            raise ValueError('Temporal feature probe unsupported for legacy baseline/no TemporalNeuron API')
        if type(expected_calls) is not int or expected_calls<1:
            raise ValueError('Positive expected call count required')
        self.handles=[];self.accumulators={};self.finished=False

    def _hook(self,name,layout):
        def hook(module,args,output):
            if (len(args)!=4 or type(args[3]) is not int or args[3] not in (0,1)
                    or not isinstance(output,torch.Tensor) or output.ndim<4):
                raise ValueError('Expected exact TemporalNeuron positional view/tensor API')
            if self.require_cuda and output.device.type!='cuda':
                raise RuntimeError('Real feature summaries must stay on allocated GPU during forward')
            value=output.detach()
            view=args[3];gate=layout['role']=='global_channel_gate'
            extent,expected=valid_extent(self.input_shape,layout['stride'])
            if gate:
                if value.shape[-2:]!=(1,1):
                    raise ValueError('Global gate must have1x1 spatial support')
                extent=(1,1)
            elif tuple(value.shape[-2:])!=expected:
                raise ValueError('Neuron feature shape disagrees with exact pad16/stride geometry: '+name)
            value=value.reshape(value.shape[0],-1,*value.shape[-2:])[...,:extent[0],:extent[1]].float()
            variance,mean=torch.var_mean(value,dim=(-2,-1),correction=0)
            mean64=mean.double();variance64=variance.double()
            key=(name,view)
            if key not in self.accumulators:
                self.accumulators[key]=dict(calls=0,stride=layout['stride'],role=layout['role'],
                    spatial_extent=extent,channel_shape=tuple(mean.shape),
                    mean_sum=torch.zeros_like(mean64),mean_square_sum=torch.zeros_like(mean64),
                    variance_sum=torch.zeros_like(variance64),
                    constant_channel_time_count=mean64.new_zeros(()),
                    minimum=value.amin(),maximum=value.amax())
            state=self.accumulators[key]
            if tuple(mean.shape)!=state['channel_shape'] or extent!=state['spatial_extent']:
                raise ValueError('Temporal neuron shape changed within one presentation')
            state['calls']+=1
            state['mean_sum'].add_(mean64)
            state['mean_square_sum'].add_(mean64.square())
            state['variance_sum'].add_(variance64)
            state['constant_channel_time_count'].add_((variance<=1e-12).double().sum())
            state['minimum']=torch.minimum(state['minimum'],value.amin())
            state['maximum']=torch.maximum(state['maximum'],value.amax())
            # Never return telemetry/output here: non-None replaces forward output.
            return None
        return hook

    def __enter__(self):
        if self.handles or self.finished:
            raise RuntimeError('A probe is single-use per selected correct-right presentation')
        try:
            for name,(module,layout) in self.nodes.items():
                self.handles.append(module.register_forward_hook(self._hook(name,layout)))
        except BaseException:
            self._remove()
            raise
        return self

    def _remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]

    def __exit__(self,*args):
        self._remove();self.finished=True
        return False

    def serialize(self):
        """One compact tensor transfer after the complete forward, never per bin."""
        if not self.finished or self.handles:
            raise RuntimeError('Serialize only after complete forward and reliable hook removal')
        expected={(name,view) for name in self.nodes for view in (0,1)}
        if set(self.accumulators)!=expected or any(s['calls']!=self.expected_calls for s in self.accumulators.values()):
            raise ValueError('Every registered node/view must have exactly the full50 temporal calls')
        records=[];values=[]
        for (name,view),state in sorted(self.accumulators.items()):
            calls=state['calls'];means=state['mean_sum']/calls
            spatial_var=state['variance_sum']/calls
            temporal_var=(state['mean_square_sum']/calls-means.square()).clamp_min(0)
            assessable=state['role']=='spatial_feature' and math.prod(state['spatial_extent'])>1
            records.append(dict(node=name,view=view,role=state['role'],stride=state['stride'],calls=calls,
                activation_kind='binary_spike' if self.nodes[name][0].spiking else 'continuous_sigmoid_counterpart',
                valid_spatial_shape=list(state['spatial_extent']),batch_channels=list(state['channel_shape']),
                spatial_variance_assessable=assessable))
            values.append(torch.stack((means.mean(),state['minimum'].double(),state['maximum'].double(),
                means.amin(),means.amax(),spatial_var.mean(),(spatial_var<=1e-12).double().mean(),
                state['constant_channel_time_count']/(calls*means.numel()),temporal_var.mean())))
        packed=torch.stack(values).detach().cpu().tolist()
        names=('activation_mean','activation_min','activation_max','time_mean_channel_activity_min',
               'time_mean_channel_activity_max','mean_within_spatial_channel_variance',
               'spatially_constant_channel_fraction','spatially_constant_channel_time_fraction',
               'temporal_variance_of_channel_spatial_means')
        for record,row in zip(records,packed):
            record.update({key:value if math.isfinite(value) else None for key,value in zip(names,row)})
            record['finite_summaries']=all(math.isfinite(value) for value in row)
            if not record['spatial_variance_assessable']:
                for key in ('mean_within_spatial_channel_variance','spatially_constant_channel_fraction','spatially_constant_channel_time_fraction'):
                    record[key]=None
        self.accumulators.clear() # Release device summaries after serialization.
        return dict(input_shape=list(self.input_shape),padded_shape=list(self.padded_shape),nodes=records,
            channel_constant_definition='time_mean_within_spatial_variance<=1e-12;fraction_over_batch_channels',
            channel_time_constant_definition='within_spatial_variance<=1e-12;fraction_over_time_batch_channels',
            temporal_variance_definition='population_variance_over50_spatial_channel_means_then_mean_over_batch_channels',
            padding_support='ceil(source/stride),matching_maxpooled_any_original_pixel_support;partial_boundary_cells_retained',
            interpretation='descriptive_activity_only;nonzero_mean_or_variance_does_not_establish_informative_stereo_features')


def output_information(output,input_shape):
    """API-only correct-forward statistics on fixed in-frame spatial support.

    Match statistics are existing max-probability/normalized entropy, not an
    occlusion estimate or calibrated confidence. Main=upsampled aux4+residual
    follows the exact pad16 then crop decoder geometry. No GT is read here.
    """
    h,w=map(int,input_shape);ph,pw=padded_shape(input_shape)
    main=output['disparity'].detach()
    if main.ndim!=4 or main.shape[1]!=1 or main.shape[-2:]!=(h,w):
        raise ValueError('Unaugmented main disparity required')
    auxiliary=output['aux_disparities_native'][4].detach()
    if auxiliary.shape[-2:]!=(ph//4,pw//4):
        raise ValueError('Stride4 matcher layout differs from exact padded decoder')
    upsampled=F.interpolate(auxiliary,size=(ph,pw),mode='bilinear',align_corners=False)[...,:h,:w]
    branches=dict(main=main,matcher_upsampled_aux4=upsampled,residual_main_minus_matcher=main-upsampled)
    metadata=[];values=[]
    def add(group,name,tensor):
        value=tensor.detach().double()
        variance,mean=torch.var_mean(value,dim=(-2,-1),correction=0)
        metadata.append(dict(group=group,name=name,spatial_shape=list(value.shape[-2:])))
        values.append(torch.stack((mean.mean(),variance.sqrt().mean(),value.amin(),value.amax())))
    for name,value in branches.items():
        add('disparity_branches',name,value)
    for stride in (4,8,16):
        extent,expected=valid_extent(input_shape,stride)
        for name in ('max_probability','normalized_entropy'):
            value=output['match_confidence_native'][stride][name]
            if value.ndim!=4 or value.shape[1]!=1 or tuple(value.shape[-2:])!=expected:
                raise ValueError('Matcher confidence/entropy scale geometry mismatch')
            add('matcher',str(stride)+'/'+name,value[...,:extent[0],:extent[1]])
    packed=torch.stack(values).detach().cpu().tolist()
    result=dict(input_shape=[h,w],padded_shape=[ph,pw],disparity_branches={},matcher={},
        support='all_original_input_pixels_for_branches;ceil_source_extent_at_matcher_scale;no_GT_or_confidence_mask',
        interpretation='descriptive_only;matcher_probability_is_not_occlusion_or_information_proof')
    for meta,row in zip(metadata,packed):
        result[meta['group']][meta['name']]=dict(spatial_shape=meta['spatial_shape'],
            **{key:value if math.isfinite(value) else None for key,value in zip(('spatial_mean','spatial_std','minimum','maximum'),row)},
            finite_summaries=all(math.isfinite(value) for value in row))
    return result


if __name__=='__main__':
    print(json.dumps(probe_contract(),indent=2))
