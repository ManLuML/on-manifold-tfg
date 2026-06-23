"""Tests for guidance space calibration utilities."""

import pytest

from jit_tfg.tfg.calibration import (
    compute_equivalent_rho_mu,
    compute_guidance_ratio_profile,
)


class TestGuidanceRatioProfile:
    """Tests for compute_guidance_ratio_profile."""

    def test_returns_expected_keys(self):
        """Result should contain num_steps, t_eps, lambda_mode, steps, cumulative."""
        result = compute_guidance_ratio_profile(10, 0.05, "flow_guidance")
        assert "num_steps" in result
        assert "t_eps" in result
        assert "lambda_mode" in result
        assert "steps" in result
        assert "cumulative" in result
        assert len(result["steps"]) == 10

    def test_x_space_larger_than_v_space(self):
        """x-space cumulative should be larger than v-space for both deltas."""
        result = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        cum = result["cumulative"]
        assert cum["x_delta_t_total"] > cum["v_delta_t_total"]
        assert cum["x_delta_0_total"] > cum["v_delta_0_total"]

    def test_identity_lambda_smaller_v_delta_t(self):
        """Identity lambda should produce smaller v-space delta_t than flow_guidance."""
        fg = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        id_ = compute_guidance_ratio_profile(50, 0.05, "identity")
        # flow_guidance amplifies early steps where (1-t)/t > 1, so v_delta_t is larger
        assert fg["cumulative"]["v_delta_t_total"] > id_["cumulative"]["v_delta_t_total"]

    def test_identity_larger_ratio_than_flow_guidance(self):
        """Identity lambda should have larger x/v ratio than flow_guidance for delta_t."""
        fg = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        id_ = compute_guidance_ratio_profile(50, 0.05, "identity")
        assert id_["cumulative"]["ratio_delta_t"] > fg["cumulative"]["ratio_delta_t"]

    def test_delta_0_ratio_same_regardless_of_lambda(self):
        """delta_0 ratio should be identical for flow_guidance and identity (lambda not applied)."""
        fg = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        id_ = compute_guidance_ratio_profile(50, 0.05, "identity")
        assert abs(fg["cumulative"]["ratio_delta_0"] - id_["cumulative"]["ratio_delta_0"]) < 0.01

    def test_delta_t_and_delta_0_ratios_differ(self):
        """delta_t and delta_0 ratios should differ (lambda only on delta_t)."""
        result = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        cum = result["cumulative"]
        assert abs(cum["ratio_delta_t"] - cum["ratio_delta_0"]) > 1.0

    def test_more_steps_v_space_converges(self):
        """v-space cumulative should converge as steps increase (integral)."""
        r50 = compute_guidance_ratio_profile(50, 0.05, "flow_guidance")
        r100 = compute_guidance_ratio_profile(100, 0.05, "flow_guidance")
        # v-space delta_0 converges to integral: Σ dt should be ~same (total time interval)
        assert abs(r50["cumulative"]["v_delta_0_total"] - r100["cumulative"]["v_delta_0_total"]) < 0.01


class TestEquivalentRhoMu:
    """Tests for compute_equivalent_rho_mu."""

    def test_equivalent_values_positive(self):
        """Equivalent v-space values should be positive."""
        equiv = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05)
        assert equiv["v_rho_equivalent"] > 0
        assert equiv["v_mu_equivalent"] > 0

    def test_v_values_larger_than_x(self):
        """v-space equivalents should be much larger than x-space values."""
        equiv = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05)
        assert equiv["v_rho_equivalent"] > equiv["x_rho"] * 10
        assert equiv["v_mu_equivalent"] > equiv["x_mu"] * 10

    def test_identity_lambda_gives_larger_v_rho(self):
        """Identity lambda should require larger v-space rho than flow_guidance."""
        fg = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05, "flow_guidance")
        id_ = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05, "identity")
        assert id_["v_rho_equivalent"] > fg["v_rho_equivalent"]

    def test_delta_0_equivalent_same_for_both_lambdas(self):
        """delta_0 equivalent should be same regardless of lambda_mode."""
        fg = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05, "flow_guidance")
        id_ = compute_equivalent_rho_mu(0.5, 0.5, 50, 0.05, "identity")
        assert abs(id_["v_mu_equivalent"] - fg["v_mu_equivalent"]) < 0.01

    def test_zero_rho_gives_zero_equivalent(self):
        """Zero x-space rho should give zero v-space rho."""
        equiv = compute_equivalent_rho_mu(0.0, 0.5, 50, 0.05)
        assert equiv["v_rho_equivalent"] == 0.0

    def test_known_ratios(self):
        """Verify approximate known ratios for N=50, t_eps=0.05.

        These match actual sampler behavior (linspace(0,1) with t_eps clamping).
        flow_guidance: ~16x for delta_t due to clamped lambda at early steps.
        identity: ~52x for delta_t (no lambda amplification at early steps).
        delta_0: ~25x for both (lambda not applied to delta_0).
        """
        # flow_guidance: ~16x for delta_t, ~25x for delta_0
        fg = compute_equivalent_rho_mu(1.0, 1.0, 50, 0.05, "flow_guidance")
        assert 13 < fg["ratio_delta_t"] < 20
        assert 22 < fg["ratio_delta_0"] < 30

        # identity: ~52x for delta_t, same ~25x for delta_0
        id_ = compute_equivalent_rho_mu(1.0, 1.0, 50, 0.05, "identity")
        assert 45 < id_["ratio_delta_t"] < 60
        assert 22 < id_["ratio_delta_0"] < 30
