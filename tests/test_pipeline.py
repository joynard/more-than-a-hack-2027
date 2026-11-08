"""Automated unit and integration test suite for the Ball Possession Pipeline."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games
import starter.load_smart_tagging as smart_tagging
import evaluation.possession_metrics as pm
from src.pipeline import (
    interpolate_inplay_ball,
    viterbi_decode_block,
    prune_block,
    build_emissions,
    H1_START, H1_END, H2_START, H2_END, H2_KICKOFF, TOTAL_FRAMES
)


class TestPipeline(unittest.TestCase):

    def test_ball_interpolation_in_bounds(self):
        """Short gaps in-bounds under speed threshold must be interpolated."""
        arr = np.full((10, 2), np.nan, dtype=np.float32)
        arr[0] = [10.0, 10.0]
        arr[3] = [11.0, 11.0]  # gap of 2 frames, speed ~ 1.4m / 0.12s ~ 11.8 m/s <= 35
        filled, is_interp = interpolate_inplay_ball(arr, max_gap=5, max_speed=35.0, fps=25.0)

        self.assertTrue(is_interp[1])
        self.assertTrue(is_interp[2])
        self.assertFalse(is_interp[0])
        self.assertFalse(is_interp[3])
        self.assertFalse(np.isnan(filled[1, 0]))
        self.assertAlmostEqual(filled[1, 0], 10.333333, places=4)

    def test_ball_interpolation_out_of_bounds_guard(self):
        """Gaps where an endpoint is outside pitch must NEVER be interpolated."""
        arr = np.full((10, 2), np.nan, dtype=np.float32)
        arr[0] = [10.0, 10.0]
        arr[3] = [110.0, 10.0]  # x = 110.0 is out of bounds (> 105.3)
        filled, is_interp = interpolate_inplay_ball(arr, max_gap=5, max_speed=35.0, fps=25.0)

        self.assertFalse(np.any(is_interp))
        self.assertTrue(np.isnan(filled[1, 0]))

    def test_viterbi_deflection_suppression(self):
        """Transient 2-frame flip must be suppressed by the sticky transition matrix."""
        T = 100
        log_e = np.zeros((T, 3), dtype=np.float32)
        # 0..100 is Home (H)
        log_e[:, 0] = 3.0
        # Inject 2-frame noise of Away (A) at 45..47
        log_e[45:47, 0] = -5.0
        log_e[45:47, 1] = 5.0

        path = viterbi_decode_block(log_e, pen_ha=40.0, pen_d=20.0)
        self.assertTrue(np.all(path == 0), f"Noise was not suppressed: {path[43:50]}")

    def test_prune_block_dead_duration(self):
        """Dead ball spells shorter than min_dead_len (80 frames) must be absorbed."""
        n = 200
        path = np.zeros(n, dtype=np.int8)  # all H
        path[50:80] = pm.D  # 30 frames of D (< 80)

        vis = np.ones(n, dtype=bool)
        h1 = np.full(n, 1.0, dtype=np.float32)
        a1 = np.full(n, 5.0, dtype=np.float32)
        spd = np.full(n, 2.0, dtype=np.float32)

        cleaned = prune_block(path, vis, h1, a1, spd, min_dead_len=80, min_alive_len=25)
        self.assertTrue(np.all(cleaned == pm.H), "Short D spell was not absorbed into H!")

    def test_occlusion_turnover_snap(self):
        """Turnover happening during ball occlusion must snap to first visible frame."""
        n = 100
        path = np.zeros(n, dtype=np.int8)  # H
        path[50:] = pm.A                   # Turnover at 50

        # Ball invisible from frame 40 to 65
        vis = np.ones(n, dtype=bool)
        vis[40:65] = False

        # Team H controls 0..50, Team A controls 50..100
        h1 = np.full(n, 5.0, dtype=np.float32)
        h1[:50] = 1.0
        a1 = np.full(n, 5.0, dtype=np.float32)
        a1[50:] = 1.0
        spd = np.full(n, 2.0, dtype=np.float32)

        cleaned = prune_block(path, vis, h1, a1, spd, min_dead_len=80, min_alive_len=10, min_ctrl_frames=5)
        # Should stay H until frame 65 (first visible frame of A's possession)
        self.assertTrue(np.all(cleaned[50:65] == pm.H), f"Turnover did not snap across occlusion: {cleaned[48:70]}")
        self.assertEqual(cleaned[65], pm.A)

    def test_second_half_kickoff_trap(self):
        """Verify second-half kickoff frame boundaries."""
        self.assertEqual(H2_START, 110808)
        self.assertEqual(H2_KICKOFF, 113742)
        self.assertEqual(H2_KICKOFF - H2_START, 2934)

    def test_uncontested_loose_ball_deadness(self):
        """When ball is on-pitch but both teams are > 4.5m and ball is slow, D is boosted."""
        n = TOTAL_FRAMES
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
        ball_xy[500:600] = [50.0, 30.0]  # center of pitch
        h1 = np.full(n, 99.0, dtype=np.float32)
        h2 = np.full(n, 99.0, dtype=np.float32)
        a1 = np.full(n, 99.0, dtype=np.float32)
        a2 = np.full(n, 99.0, dtype=np.float32)
        h1[500:600] = 6.0  # both teams distant
        a1[500:600] = 6.0
        b_spd = np.zeros(n, dtype=np.float32)
        b_spd[500:600] = 4.0  # rolling slowly
        p_mean = np.full(n, 2.0, dtype=np.float32)
        p_p90 = np.full(n, 2.5, dtype=np.float32)
        is_interp = np.zeros(n, dtype=bool)

        log_e = build_emissions(ball_xy, h1, h2, a1, a2, b_spd, p_mean, p_p90, [], is_interp)
        # Frames 500..600 should favor Dead Ball (D=state 2) over H and A
        self.assertGreater(log_e[550, 2], log_e[550, 0])
        self.assertGreater(log_e[550, 2], log_e[550, 1])

    def test_high_tempo_vis_guard(self):
        """High running tempo while ball is NaN must NOT penalize Dead Ball D."""
        n = TOTAL_FRAMES
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)  # ball missing
        h1 = np.full(n, 99.0, dtype=np.float32)
        h2 = np.full(n, 99.0, dtype=np.float32)
        a1 = np.full(n, 99.0, dtype=np.float32)
        a2 = np.full(n, 99.0, dtype=np.float32)
        b_spd = np.zeros(n, dtype=np.float32)
        p_mean = np.full(n, 3.5, dtype=np.float32)  # high running speed
        p_p90 = np.full(n, 5.0, dtype=np.float32)
        is_interp = np.zeros(n, dtype=bool)

        log_e = build_emissions(ball_xy, h1, h2, a1, a2, b_spd, p_mean, p_p90, [], is_interp)
        # Because ball is missing, high tempo should NOT give D a -3.5 penalty
        self.assertGreaterEqual(log_e[550, 2], 0.0)

    def test_dead_ball_retrieval_guard(self):
        """A lone player walking up to retrieve a slow ball while collective tempo is stopped (<1.32) must stay D."""
        n = TOTAL_FRAMES
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
        ball_xy[500:600] = [9.0, 60.0]
        h1 = np.full(n, 99.0, dtype=np.float32)
        h2 = np.full(n, 99.0, dtype=np.float32)
        a1 = np.full(n, 99.0, dtype=np.float32)
        a2 = np.full(n, 99.0, dtype=np.float32)
        h1[500:600] = 0.5   # Home player picking up ball
        a1[500:600] = 35.0  # Away team > 20m away
        b_spd = np.zeros(n, dtype=np.float32)
        b_spd[500:600] = 2.0  # Ball moving < 4.0 m/s
        p_mean = np.full(n, 1.25, dtype=np.float32)  # Collective tempo stopped (< 1.32)
        p_p90 = np.full(n, 2.50, dtype=np.float32)   # 2 players jogging (> 2.15)
        is_interp = np.zeros(n, dtype=bool)

        log_e = build_emissions(ball_xy, h1, h2, a1, a2, b_spd, p_mean, p_p90, [], is_interp)
        # Retrieval guard must suppress h_ctrl and boost D (state 2 > state 0)
        self.assertGreater(log_e[550, 2], log_e[550, 0])
        self.assertEqual(log_e[550, 0], 0.0)

    def test_pass_flight_turnover_and_final_kick_snap(self):
        """Turnovers during high-speed pass flight (>10 m/s) snap forward; accelerating kicks into D hold possession."""
        n = 250
        path = np.full(n, pm.H, dtype=np.int8)
        path[50:150] = pm.A
        path[150:] = pm.D
        vis = np.ones(n, dtype=bool)
        h1 = np.full(n, 1.0, dtype=np.float32)
        h1[60:] = 8.0
        a1 = np.full(n, 5.0, dtype=np.float32)
        a1[60:150] = 1.0
        spd = np.full(n, 3.0, dtype=np.float32)
        # Frames 50..60 are in high-speed pass flight (spd = 14.0 m/s) before A receives at frame 60
        spd[50:60] = 14.0
        # Frames 150..165 are an accelerating kick into D (spd rises from 6.0 to 13.0, then drops below 10.0 at 165)
        spd[150:158] = np.linspace(6.0, 13.0, 8)
        spd[158:165] = 11.5
        spd[165:] = 4.0

        cleaned = prune_block(path, vis, h1, a1, spd, min_dead_len=80, min_alive_len=10, min_ctrl_frames=5)
        self.assertTrue(np.all(cleaned[50:60] == pm.H), "Turnover inside pass flight must snap to reception at 60")
        self.assertEqual(cleaned[60], pm.A)
        self.assertTrue(np.all(cleaned[150:165] == pm.A), "Accelerating kick into D must hold possession until flight decelerates")
        self.assertEqual(cleaned[165], pm.D)


if __name__ == "__main__":
    unittest.main()
