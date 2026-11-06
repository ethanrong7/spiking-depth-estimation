"""Parameter-identical repeated application of the sole native stereo MRT."""
from types import MethodType

def apply_tied_mrt(model,passes=2):
    transformer=model.transformer
    assert len(transformer.uformer_list)==1 and passes in (1,2)
    transformer.tied_mrt_passes=passes
    def repeated(module,z0,z1,z2,z3):
        for _ in range(module.tied_mrt_passes):
            z0,z1,z2,z3=module.uformer_list[0](z0,z1,z2,z3)
        return z0.contiguous()
    transformer.forward=MethodType(repeated,transformer)
    return model
