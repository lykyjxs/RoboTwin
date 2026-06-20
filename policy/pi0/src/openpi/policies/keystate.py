"""KeyState data transform.

Maps the raw per-frame KeyState labels carried alongside the observation
(`keystate.{next_checkpoint_type, h_entry, semantic_phase}`, written by
`scripts/process_data.py` and registered as LeRobot features by
`examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`) into the fields the
model expects on `model.Observation`: `keystate_type` / `keystate_h_entry` / `keystate_phase`.

Stage 2 may additionally carry `keystate.z_entry_descriptor`, a checkpoint-entry
Key-state latent target. The current main backend extracts it from frozen Pi0 action-expert
hidden and projects it to 64D; older bootstrap/prefix backends use the same field name.

Two responsibilities live here (and nowhere else, so the contract is in one place):

1. Bucket the raw integer distance-to-window-entry `h_entry` into a bin index,
   matching `Pi0Config.horizon_upper_edges` ("h < edge -> that bin") EXACTLY. The invalid
   sentinel `h_entry == -1` (frames with no next/current checkpoint window) is preserved as
   `-1`; the model's `_keystate_losses` reads `keystate_h_entry >= 0` as its horizon-valid
   mask, so the sentinel must survive untouched (do NOT clip it into bin 0).
2. Pass `next_checkpoint_type` through as `keystate_type` and `semantic_phase` as
   `keystate_phase` (float, for BCE), squeezing the trailing singleton dim that the
   `(1,)`-shaped LeRobot scalar features carry.
3. Pass optional `z_entry_descriptor` through as `keystate_z_entry_descriptor` without
   bucketing or normalization; the generator controls descriptor scale.

This transform is a no-op when `keystate` is absent (baseline configs / inference),
so existing pipelines are unaffected.
"""

import dataclasses

import numpy as np

from openpi import transforms


def bucket_h_entry(h_entry: np.ndarray, upper_edges: tuple[int, ...]) -> np.ndarray:
    """Distance-to-window-entry binning, matching Pi0Config.horizon_upper_edges semantics.

    "h < edge -> that bin": with edges (1, 4, 7, 11, 21, 51) ->
      h==0 -> 0 | 1<=h<4 -> 1 | 4<=h<7 -> 2 | 7<=h<11 -> 3 |
      11<=h<21 -> 4 | 21<=h<51 -> 5 | else -> 6.
    i.e. num_bins = len(upper_edges) + 1. The invalid sentinel (h_entry < 0) is passed
    through as -1 so the model can mask it out (it is never a valid bin index).
    """
    h_entry = np.asarray(h_entry)
    # np.searchsorted(edges, h, side="right") gives the count of edges <= h, which is
    # exactly the "first edge strictly greater than h" bin index under the "h < edge" rule.
    bins = np.searchsorted(np.asarray(upper_edges), h_entry, side="right").astype(np.int32)
    return np.where(h_entry < 0, np.int32(-1), bins)


@dataclasses.dataclass(frozen=True)
class KeyStateInputs(transforms.DataTransformFn):
    """Derive model KeyState labels from the raw per-frame keystate sub-dict.

    Push this AFTER AlohaInputs in the data-transform group (AlohaInputs forwards the
    raw `keystate` dict untouched; this transform consumes it and emits the model keys).
    """

    # Must match Pi0Config.horizon_upper_edges for the buckets to line up with the head.
    horizon_upper_edges: tuple[int, ...] = (1, 4, 7, 11, 21, 51)

    def __call__(self, data: dict) -> dict:
        ks = data.get("keystate")
        if ks is None:
            # baseline / inference: nothing to do.
            return data

        # The LeRobot scalar features are registered as shape (1,); squeeze the trailing
        # singleton so keystate_type/keystate_h_entry are per-sample scalars (the model treats them
        # as [*b] ints). semantic_phase is a real (num_phase,) vector -> left as-is.
        next_type = _squeeze_scalar(ks["next_checkpoint_type"])
        h_entry_raw = _squeeze_scalar(ks["h_entry"])

        data["keystate_type"] = next_type.astype(np.int32)
        data["keystate_h_entry"] = bucket_h_entry(h_entry_raw, self.horizon_upper_edges)
        # multi-label targets for BCE -> float.
        data["keystate_phase"] = np.asarray(ks["semantic_phase"]).astype(np.float32)
        if "z_entry_descriptor" in ks:
            data["keystate_z_entry_descriptor"] = np.asarray(ks["z_entry_descriptor"]).astype(np.float32)

        # consumed: drop the raw sub-dict so it does not leak into the model input dict.
        data.pop("keystate", None)
        return data


def _squeeze_scalar(x) -> np.ndarray:
    """Drop a trailing singleton dim (the registered (1,) scalar-feature shape), if present."""
    x = np.asarray(x)
    if x.ndim >= 1 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
    return x
