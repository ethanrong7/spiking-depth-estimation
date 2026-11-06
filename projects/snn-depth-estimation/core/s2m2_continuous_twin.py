"""Parameter-identical E1 transfer substitution at v2b's retained coded sites."""
from types import MethodType
from core.s2m2_spiking import SpikeProjection, SpikeHidden

def projection(self,x):
    out=self.linear(x).clamp(-1,1)
    self.last_rate=out.detach().abs().mean()
    return out

def hidden(self,x):
    out=x.clamp(-1,1)
    self.last_rate=out.detach().abs().mean()
    return out

def apply_continuous_codes(model):
    sites=[]
    for name,module in model.named_modules():
        if isinstance(module,SpikeProjection):module.forward=MethodType(projection,module);sites.append(name)
        elif isinstance(module,SpikeHidden):module.forward=MethodType(hidden,module);sites.append(name)
    assert sites and 'disp_init.q' in sites and 'disp_init.k' in sites
    model.continuous_sites=sites
    return model
