"""Pooled-valid effective batch2, with exactly one clip/optimizer/scheduler step."""
import math
import torch


def sequence_sum(predictions,target,valid):
 if len(predictions)!=16 or any(not torch.isfinite(p).all() for p in predictions):raise ValueError('exactly16finite predictions required')
 mask=valid.unsqueeze(1) if valid.ndim==3 else valid
 mask=mask&torch.isfinite(target)
 return sum(.9**(15-i)*(prediction-target).abs()[mask].sum() for i,prediction in enumerate(predictions))


def train_pair(model,name,items,optimizer,scheduler,forward_fn,reset_fn,clip_fn=torch.nn.utils.clip_grad_norm_):
 if len(items)!=2:raise ValueError('effectivebatch exactly2')
 counts=[int(item['valid'].sum()) for item in items];total=sum(counts)
 if total<=0:raise RuntimeError('effectivebatch has no valid source GT')
 model.train();model.freeze_bn();optimizer.zero_grad(set_to_none=True);loss_value=0.;error_sum=0.
 for item in items:
  left,right,target,valid=[item[k].unsqueeze(0) for k in ['left','right','target','valid']]
  preds=forward_fn(model,name,left,right,16,True)
  loss=sequence_sum(preds,target,valid)/total
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite disparityloss')
  loss.backward();loss_value+=float(loss.detach());error_sum+=float((preds[-1]-target).abs()[valid[:,None]].sum().detach())
  # Free the unrolled graph and ALIF memories before the next independent sample.
  reset_fn(model);del preds,loss,left,right,target,valid
 gradient=float(clip_fn(model.parameters(),1.))
 if not math.isfinite(gradient) or gradient<=0:raise FloatingPointError('invalid accumulated gradient')
 used_lr=float(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
 return dict(loss=loss_value,disparity_loss=loss_value,neuron_loss=0.,train_epe=error_sum/total,gradient_norm=gradient,used_lr=used_lr,lr=float(scheduler.get_last_lr()[0]),valid_pixels=total,microbatch_valid_pixels=counts)
