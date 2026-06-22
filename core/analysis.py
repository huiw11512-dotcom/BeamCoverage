from __future__ import annotations

from core.element_pattern import ElementPattern, make_element_pattern, scan_loss_db_for_direction
from core.geometry import BeamParams, DerivedParams, derive_params


def derive_params_with_element(params: BeamParams) -> tuple[BeamParams, DerivedParams, ElementPattern]:
    """Return sanitized params, derived geometry, and the active element model.

    ``derive_params`` intentionally stays geometry-only so it can be reused by
    lightweight validation paths. This helper is the core calculation entry for
    workflows that also need element-pattern-aware automatic results such as
    scan loss, especially when a CSV element pattern is active.
    """

    sanitized, derived = derive_params(params)
    elem = make_element_pattern(sanitized, derived.wavelength_m)
    derived.scan_loss_db = scan_loss_db_for_direction(derived.u0, derived.v0, derived.w0, elem)
    return sanitized, derived, elem
