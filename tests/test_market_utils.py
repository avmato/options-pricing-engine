import numpy as np
import pytest

from src.market_utils import (
    call_intrinsic_value,
    log_moneyness,
    moneyness,
    put_intrinsic_value,
    years_to_expiry,
)


def test_call_intrinsic_value_scalar() -> None:
    assert call_intrinsic_value(100.0, 80.0) == 20.0
    assert call_intrinsic_value(100.0, 120.0) == 0.0


def test_put_intrinsic_value_scalar() -> None:
    assert put_intrinsic_value(100.0, 120.0) == 20.0
    assert put_intrinsic_value(100.0, 80.0) == 0.0


def test_intrinsic_values_vectorized() -> None:
    strikes = np.array([80.0, 100.0, 120.0])

    call_values = call_intrinsic_value(100.0, strikes)
    put_values = put_intrinsic_value(100.0, strikes)

    np.testing.assert_allclose(
        call_values,
        np.array([20.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        put_values,
        np.array([0.0, 0.0, 20.0]),
    )


def test_moneyness() -> None:
    result = moneyness(
        100.0,
        np.array([80.0, 100.0, 125.0]),
    )

    np.testing.assert_allclose(
        result,
        np.array([1.25, 1.0, 0.8]),
    )


def test_log_moneyness_at_the_money() -> None:
    assert log_moneyness(100.0, 100.0) == pytest.approx(0.0)


def test_years_to_expiry() -> None:
    result = years_to_expiry(
        np.array([30.0, 365.0]),
    )

    np.testing.assert_allclose(
        result,
        np.array([30.0 / 365.0, 1.0]),
    )


def test_moneyness_rejects_nonpositive_strike() -> None:
    with pytest.raises(ValueError):
        moneyness(100.0, 0.0)


def test_log_moneyness_rejects_nonpositive_spot() -> None:
    with pytest.raises(ValueError):
        log_moneyness(0.0, 100.0)


def test_years_to_expiry_rejects_negative_days() -> None:
    with pytest.raises(ValueError):
        years_to_expiry(-1.0)