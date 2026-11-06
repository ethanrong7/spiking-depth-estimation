"""State-identical linear-QKV controls for the historical A/B placements."""
from core import spiking_attention_placement as historical


class LinearCodes(historical.Codes):
    def forward(self, x):
        z = self.norm(x)
        return tuple(layer(z) for layer in (self.q, self.k, self.v))


def install(model, arm):
    if arm not in ('A', 'B'):
        raise ValueError(arm)
    model = historical.install(model, arm)
    # No allocation, initialization or state-key change; only encoding changes.
    model.placement.codes.__class__ = LinearCodes
    return model
