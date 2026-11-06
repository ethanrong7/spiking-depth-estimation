"""Exact global-batch loss statistics and gradient replay for T50 feasibility.

Run the first model pass under ``torch.no_grad()``, call ``accumulate`` for
every physical microbatch, and reset the stateful model after each forward.
During the second pass, call ``replay_loss(...).backward()`` immediately for
each recreated microbatch, again resetting after each, but do not update model
parameters until the complete effective batch has replayed.

The replay rate term is a linear gradient surrogate. Its numerical value is
not the source objective; report the detached ``stats.true_loss`` instead.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from core.utils.utils import InputPadder


REFINEMENTS = 16
PREDICTION_WEIGHTS = tuple(0.9 ** (15 - index) for index in range(REFINEMENTS))
MAX_DISP = 700
TARGET_RATE = 0.05
LAMBDA_RATE = 1e-4
LAMBDA_VOLTAGE = 1e-3


@dataclass(frozen=True)
class ExactBatchLossStats:
    """Detached sufficient statistics for one effective global batch."""

    disparity_numerator: torch.Tensor
    valid_pixel_count: int
    spike_sum: torch.Tensor
    spike_batch_refinement_count: int
    voltage_square_sum: torch.Tensor
    voltage_element_count: int

    @property
    def global_rate(self) -> torch.Tensor:
        return self.spike_sum / self.spike_batch_refinement_count

    @property
    def true_disparity_loss(self) -> torch.Tensor:
        return self.disparity_numerator / self.valid_pixel_count

    @property
    def true_rate_loss(self) -> torch.Tensor:
        return LAMBDA_RATE * ((self.global_rate - TARGET_RATE) ** 2).sum()

    @property
    def true_voltage_loss(self) -> torch.Tensor:
        return LAMBDA_VOLTAGE * self.voltage_square_sum / self.voltage_element_count

    @property
    def true_loss(self) -> torch.Tensor:
        """Detached value of the unchanged full-batch source objective."""

        return self.true_disparity_loss + self.true_rate_loss + self.true_voltage_loss


@dataclass(frozen=True)
class _LocalTerms:
    disparity_numerator: torch.Tensor
    valid_pixel_count: int
    spike_sum: torch.Tensor
    spike_batch_refinement_count: int
    voltage_square_sum: torch.Tensor
    voltage_element_count: int


def _local_terms(
    s_list: Sequence[torch.Tensor],
    v_list: Sequence[torch.Tensor],
    disp_preds: Sequence[torch.Tensor],
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    spike_size: Sequence[int],
) -> _LocalTerms:
    lengths = (len(s_list), len(v_list), len(disp_preds))
    if lengths != (REFINEMENTS, REFINEMENTS, REFINEMENTS):
        raise ValueError(
            "exact T50 replay requires 16 updater spikes, voltages, and predictions; "
            f"got {lengths}"
        )

    spike = torch.stack(tuple(s_list), dim=1)
    voltage = torch.stack(tuple(v_list), dim=1)
    magnitude = torch.sum(disp_gt**2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (magnitude < MAX_DISP)).unsqueeze(1)
    if valid_mask.shape != disp_gt.shape:
        raise ValueError(
            f"valid mask {tuple(valid_mask.shape)} does not match ground truth "
            f"{tuple(disp_gt.shape)}"
        )
    if torch.isinf(disp_gt[valid_mask]).any():
        raise ValueError("valid ground-truth disparity contains infinity")

    numerator = disp_gt.new_zeros(())
    padder = InputPadder(torch.Size(spike_size), divis_by=4)
    for weight, prediction in zip(PREDICTION_WEIGHTS, disp_preds):
        if not torch.isfinite(prediction).all():
            raise ValueError("disparity prediction contains a nonfinite value")
        error = (padder.unpad(prediction) - disp_gt).abs()
        if error.shape != valid_mask.shape:
            raise ValueError(
                f"unpadded prediction error {tuple(error.shape)} does not match "
                f"valid mask {tuple(valid_mask.shape)}"
            )
        numerator = numerator + weight * error[valid_mask].sum()

    return _LocalTerms(
        numerator,
        int(valid_mask.sum().item()),
        spike.sum(dim=(0, 1)),
        spike.shape[0] * spike.shape[1],
        (voltage**2).sum(),
        voltage.numel(),
    )


class ExactBatchLossAccumulator:
    """Accumulate detached source-loss statistics across microbatches."""

    def __init__(self) -> None:
        self._disparity_numerator: Optional[torch.Tensor] = None
        self._spike_sum: Optional[torch.Tensor] = None
        self._voltage_square_sum: Optional[torch.Tensor] = None
        self._valid_pixel_count = 0
        self._spike_batch_refinement_count = 0
        self._voltage_element_count = 0
        self._finalized = False

    @staticmethod
    def _add(
        total: Optional[torch.Tensor], value: torch.Tensor, name: str
    ) -> torch.Tensor:
        value = value.detach()
        if total is None:
            return value.clone()
        if (
            total.shape != value.shape
            or total.dtype != value.dtype
            or total.device != value.device
        ):
            raise ValueError(
                f"all microbatch {name} tensors must share shape, dtype, and device"
            )
        return total + value

    @torch.no_grad()
    def accumulate(
        self,
        s_list: Sequence[torch.Tensor],
        v_list: Sequence[torch.Tensor],
        disp_preds: Sequence[torch.Tensor],
        disp_gt: torch.Tensor,
        valid: torch.Tensor,
        spike_size: Sequence[int],
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot accumulate after finalize")
        terms = _local_terms(s_list, v_list, disp_preds, disp_gt, valid, spike_size)
        self._disparity_numerator = self._add(
            self._disparity_numerator, terms.disparity_numerator, "disparity"
        )
        self._spike_sum = self._add(self._spike_sum, terms.spike_sum, "spike")
        self._voltage_square_sum = self._add(
            self._voltage_square_sum, terms.voltage_square_sum, "voltage"
        )
        self._valid_pixel_count += terms.valid_pixel_count
        self._spike_batch_refinement_count += terms.spike_batch_refinement_count
        self._voltage_element_count += terms.voltage_element_count

    def finalize(self) -> ExactBatchLossStats:
        if self._finalized:
            raise RuntimeError("accumulator has already been finalized")
        if any(
            value is None
            for value in (
                self._disparity_numerator,
                self._spike_sum,
                self._voltage_square_sum,
            )
        ):
            raise RuntimeError("cannot finalize an empty accumulator")
        if not all(
            (
                self._valid_pixel_count,
                self._spike_batch_refinement_count,
                self._voltage_element_count,
            )
        ):
            raise ValueError("effective batch contains an empty loss reduction")
        self._finalized = True
        return ExactBatchLossStats(
            self._disparity_numerator,
            self._valid_pixel_count,
            self._spike_sum,
            self._spike_batch_refinement_count,
            self._voltage_square_sum,
            self._voltage_element_count,
        )


def replay_loss(
    stats: ExactBatchLossStats,
    s_list: Sequence[torch.Tensor],
    v_list: Sequence[torch.Tensor],
    disp_preds: Sequence[torch.Tensor],
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    spike_size: Sequence[int],
) -> torch.Tensor:
    """Return one microbatch scalar whose gradient matches the global loss."""

    terms = _local_terms(s_list, v_list, disp_preds, disp_gt, valid, spike_size)
    if terms.spike_sum.shape != stats.spike_sum.shape:
        raise ValueError("replay spike shape does not match first-pass statistics")

    disparity = terms.disparity_numerator / stats.valid_pixel_count
    voltage = LAMBDA_VOLTAGE * terms.voltage_square_sum / stats.voltage_element_count
    rate_coefficient = (
        2 * LAMBDA_RATE * (stats.global_rate - TARGET_RATE)
    ).detach()
    rate_gradient = (
        rate_coefficient * terms.spike_sum / stats.spike_batch_refinement_count
    ).sum()
    return disparity + voltage + rate_gradient
