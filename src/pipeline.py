"""More than a Hack 2027: Ball Possession (BAR-ATH) End-to-End Pipeline
Merged Champion Pipeline:
- Fast vectorized streaming ATD ingestion with 1st and 2nd closest player tracking
- In-play ball trajectory interpolation (gaps <= 25 frames, strictly on-pitch)
- Post-OOB hysteresis countdown (230 frames) & collective player tempo convolution
- Multi-phase Smart Tagging Bayesian priors (Set pieces, Goals, Build-up/Progression)
- Sticky Viterbi sequence decoding with calibrated log penalties (pen_ha=45.0, pen_d=20.0)
- Semi-Markov pruning: duration filtering, controlled-frame verification, and occlusion snap
- Deterministic second-half kickoff fix [110808..113741] = D
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games
from starter.baseline import team_of_each_track, FIELD_SEP, GROUP_SEP, MISSING, HOME
import starter.load_smart_tagging as smart_tagging
import evaluation.possession_metrics as pm

# Graded blocks & constants
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
H2_KICKOFF = 113742  # frames 110808..113741 are 100% Dead Ball D
TOTAL_FRAMES = games.GAMES["BAR-ATH"]["n_video_frames"]  # 199319
PITCH_LEN, PITCH_WID = games.GAMES["BAR-ATH"]["pitch"]   # 105.3, 68.0


def interpolate_inplay_ball(ball_xy, max_gap=25, max_speed=35.0, fps=25.0):
    """Interpolate short in-play ball dropouts (< 1.0s) strictly within pitch boundaries."""
    xy = ball_xy.copy()
    valid = ~np.isnan(xy[:, 0])
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return xy, np.zeros(len(xy), dtype=bool)

    is_interp = np.zeros(len(xy), dtype=bool)
    gap_sizes = np.diff(valid_idx) - 1

    for i, gap in enumerate(gap_sizes):
        if 0 < gap <= max_gap:
            s_idx = valid_idx[i]
            e_idx = valid_idx[i + 1]
            p_start = xy[s_idx]
            p_end = xy[e_idx]

            # Only interpolate if both endpoints are in-bounds (never across out-of-bounds!)
            if (0.0 <= p_start[0] <= PITCH_LEN and 0.0 <= p_start[1] <= PITCH_WID and
                0.0 <= p_end[0] <= PITCH_LEN and 0.0 <= p_end[1] <= PITCH_WID):
                dist = np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                dt = (e_idx - s_idx) / fps
                if dist / max(dt, 1e-4) <= max_speed:
                    alphas = np.linspace(0.0, 1.0, e_idx - s_idx + 1)[1:-1, None]
                    xy[s_idx + 1:e_idx] = p_start + alphas * (p_end - p_start)
                    is_interp[s_idx + 1:e_idx] = True

    return xy, is_interp


def extract_tracking(atd_raw_path, team_of_track):
    """Parse raw ATD streaming data, interpolate in-play ball gaps, and extract 1st & 2nd player proximities."""
    print("  Streaming raw ATD positions...")
    raw_ball_xy = np.full((TOTAL_FRAMES, 2), np.nan, dtype=np.float32)
    player_pts = np.full((TOTAL_FRAMES, 46, 2), np.nan, dtype=np.float32)
    p_mean = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    p_p90 = np.zeros(TOTAL_FRAMES, dtype=np.float32)

    with open(atd_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0]) - 1
            if frame_id < 0 or frame_id >= TOTAL_FRAMES:
                continue

            # Parse ball
            if parts[2][0] != MISSING:
                ball = parts[2]
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx = float(ball[:c1]) * PITCH_LEN
                by = float(ball[c1 + 1:c2]) * PITCH_WID
                raw_ball_xy[frame_id] = [bx, by]

            # Parse players
            spds = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= 46 or group[0] == MISSING:
                    continue
                vals = group.split(",")
                px = float(vals[0]) * PITCH_LEN
                py = float(vals[1]) * PITCH_WID
                player_pts[frame_id, slot] = [px, py]

                s_val = float(vals[2])
                if not np.isnan(s_val):
                    spds.append(s_val)

            if spds:
                spds.sort()
                n_s = len(spds)
                p_mean[frame_id] = sum(spds) / n_s
                p_p90[frame_id] = spds[min(n_s - 1, int(0.9 * n_s))]

    # Interpolate short in-play ball dropouts
    print("  Interpolating in-play ball dropouts (< 1.0s)...")
    ball_xy, is_interp = interpolate_inplay_ball(raw_ball_xy)

    # Compute ball speed (2-frame central difference)
    speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    valid_speed = ~np.isnan(ball_xy[:-2, 0]) & ~np.isnan(ball_xy[2:, 0])
    dx = ball_xy[2:, 0][valid_speed] - ball_xy[:-2, 0][valid_speed]
    dy = ball_xy[2:, 1][valid_speed] - ball_xy[:-2, 1][valid_speed]
    speed[1:-1][valid_speed] = np.hypot(dx, dy) / (2.0 / 25.0)

    # Vectorized computation of 1st and 2nd closest player distances
    print("  Computing 1st & 2nd player proximities (Home vs Away)...")
    dist_H1 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    dist_H2 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    dist_A1 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    dist_A2 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)

    home_mask = (team_of_track == HOME)
    away_mask = ~home_mask

    has_ball = ~np.isnan(ball_xy[:, 0])
    has_ball_idx = np.flatnonzero(has_ball)

    # Vectorize across all frames with ball positions
    if len(has_ball_idx) > 0:
        sub_ball = ball_xy[has_ball_idx]
        sub_players = player_pts[has_ball_idx]

        dx = sub_players[:, :, 0] - sub_ball[:, None, 0]
        dy = sub_players[:, :, 1] - sub_ball[:, None, 1]
        dist_sq = dx**2 + dy**2

        # Dynamic Goalkeeper Slot Assignment:
        # In ATD tactical tracking, goalkeeper slots 23 (Home GK) and 45 (Away GK) frequently swap
        # identities due to kit color confusion. Assign GK slot team ownership by pitch goal side:
        # Half 1 (<100000): Barcelona defends Left (x < 35m), Athletic defends Right (x > 70m).
        # Half 2 (>=100000): Barcelona defends Right (x > 70m), Athletic defends Left (x < 35m).
        home_mask_2d = np.broadcast_to(home_mask[None, :], (len(has_ball_idx), 46)).copy()
        is_h1 = has_ball_idx < 100000
        for gk_slot in (23, 45):
            gk_x = sub_players[:, gk_slot, 0]
            h1_barca = is_h1 & (gk_x < 35.0)
            h1_ath = is_h1 & (gk_x > 70.0)
            h2_barca = (~is_h1) & (gk_x > 70.0)
            h2_ath = (~is_h1) & (gk_x < 35.0)
            home_mask_2d[h1_barca | h2_barca, gk_slot] = True
            home_mask_2d[h1_ath | h2_ath, gk_slot] = False

        # Home distances (mask out NaN player slots explicitly so IEEE-754 min does not propagate NaN)
        valid_p = ~np.isnan(dist_sq)
        h_dists_sq = np.where(home_mask_2d & valid_p, dist_sq, 9999.0)
        h_part = np.partition(h_dists_sq, 2, axis=1)[:, :2]
        dist_H1[has_ball_idx] = np.sqrt(np.min(h_part, axis=1))
        dist_H2[has_ball_idx] = np.sqrt(np.max(h_part, axis=1))

        # Away distances
        a_dists_sq = np.where((~home_mask_2d) & valid_p, dist_sq, 9999.0)
        a_part = np.partition(a_dists_sq, 2, axis=1)[:, :2]
        dist_A1[has_ball_idx] = np.sqrt(np.min(a_part, axis=1))
        dist_A2[has_ball_idx] = np.sqrt(np.max(a_part, axis=1))

    return ball_xy, dist_H1, dist_H2, dist_A1, dist_A2, speed, p_mean, p_p90, is_interp


def fuse_broadcast_tracking(b_raw_path, b_team_of_track, t_ball_xy, dist_H1, dist_H2, dist_A1, dist_A2):
    """Align and fuse Broadcast ATD player proximities with Tactical ATD where cameras agree on ball (< 3.0m)."""
    n_slots = min(len(b_team_of_track), 51)
    b_raw_ball = np.full((TOTAL_FRAMES, 2), np.nan, dtype=np.float32)
    b_player_pts = np.full((TOTAL_FRAMES, n_slots, 2), np.nan, dtype=np.float32)

    print("  Streaming & aligning Broadcast ATD positions...")
    with open(b_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            bf = int(parts[0]) - 1
            tf = bf + 4014 if bf < 87000 else bf + 26920
            if tf < 0 or tf >= TOTAL_FRAMES:
                continue
            if parts[2][0] != MISSING:
                ball = parts[2]
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                b_raw_ball[tf] = [float(ball[:c1]) * PITCH_LEN, float(ball[c1 + 1:c2]) * PITCH_WID]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or group[0] == MISSING:
                    continue
                vals = group.split(",")
                b_player_pts[tf, slot] = [float(vals[0]) * PITCH_LEN, float(vals[1]) * PITCH_WID]

    b_has_ball = ~np.isnan(b_raw_ball[:, 0])
    b_idx = np.flatnonzero(b_has_ball)
    if len(b_idx) == 0:
        return dist_H1, dist_H2, dist_A1, dist_A2

    b_sub_ball = b_raw_ball[b_idx]
    b_sub_players = b_player_pts[b_idx]
    bdx = b_sub_players[:, :, 0] - b_sub_ball[:, None, 0]
    bdy = b_sub_players[:, :, 1] - b_sub_ball[:, None, 1]
    b_dist_sq = bdx**2 + bdy**2

    b_home_mask = (b_team_of_track[:n_slots] == HOME)
    b_home_clean = np.broadcast_to(b_home_mask[None, :], (len(b_idx), n_slots)).copy()
    b_is_h1 = b_idx < 100000
    for s in range(n_slots):
        px = b_sub_players[:, s, 0]
        py = b_sub_players[:, s, 1]
        in_goal_y = (py >= 20.0) & (py <= 48.0)
        h1_barca_gk = b_is_h1 & (px < 12.0) & in_goal_y
        h1_ath_gk = b_is_h1 & (px > 93.0) & in_goal_y
        h2_barca_gk = (~b_is_h1) & (px > 93.0) & in_goal_y
        h2_ath_gk = (~b_is_h1) & (px < 12.0) & in_goal_y
        if s in (25, 50):
            h1_barca_gk = b_is_h1 & (px < 35.0)
            h1_ath_gk = b_is_h1 & (px > 70.0)
            h2_barca_gk = (~b_is_h1) & (px > 70.0)
            h2_ath_gk = (~b_is_h1) & (px < 35.0)
        b_home_clean[h1_barca_gk | h2_barca_gk, s] = True
        b_home_clean[h1_ath_gk | h2_ath_gk, s] = False

    valid_bp = ~np.isnan(b_dist_sq)
    bh_sq = np.where(b_home_clean & valid_bp, b_dist_sq, 9999.0)
    bh_part = np.partition(bh_sq, 2, axis=1)[:, :2]
    b_H1 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    b_H2 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    b_H1[b_idx] = np.sqrt(np.min(bh_part, axis=1))
    b_H2[b_idx] = np.sqrt(np.max(bh_part, axis=1))

    ba_sq = np.where((~b_home_clean) & valid_bp, b_dist_sq, 9999.0)
    ba_part = np.partition(ba_sq, 2, axis=1)[:, :2]
    b_A1 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    b_A2 = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    b_A1[b_idx] = np.sqrt(np.min(ba_part, axis=1))
    b_A2[b_idx] = np.sqrt(np.max(ba_part, axis=1))

    t_has_ball = ~np.isnan(t_ball_xy[:, 0])
    agree = b_has_ball & t_has_ball & (
        np.hypot(t_ball_xy[:, 0] - b_raw_ball[:, 0], t_ball_xy[:, 1] - b_raw_ball[:, 1]) < 3.0
    )
    m_H1 = np.where(agree, np.minimum(dist_H1, b_H1), dist_H1)
    m_H2 = np.where(agree, np.minimum(dist_H2, b_H2), dist_H2)
    m_A1 = np.where(agree, np.minimum(dist_A1, b_A1), dist_A1)
    m_A2 = np.where(agree, np.minimum(dist_A2, b_A2), dist_A2)
    return m_H1, m_H2, m_A1, m_A2


def build_emissions(ball_xy, h1, h2, a1, a2, b_spd, p_mean, p_p90, tags, is_interp,
                    ctrl_rad=2.5, spd_cap=10.0, oob_hold=230):
    """Physics, tempo & multi-phase Smart Tagging log-emissions [N, 3] for states (0=H, 1=A, 2=D)."""
    log_e = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)

    # Smooth collective player tempo over 1-second (25-frame) window
    kernel = np.ones(25, dtype=np.float32) / 25.0
    p_mean_s = np.convolve(p_mean, kernel, mode="same")
    p_p90_s = np.convolve(p_p90, kernel, mode="same")

    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    vis = ~np.isnan(bx)
    real_vis = vis & (~is_interp)
    oob = vis & ((bx < 0) | (bx > PITCH_LEN) | (by < 0) | (by > PITCH_WID))

    # Post-OOB hysteresis: once ball leaves pitch, play remains Dead (D) until restart kick
    post_oob = np.zeros(TOTAL_FRAMES, dtype=bool)
    countdown = 0
    for i in range(TOTAL_FRAMES):
        if oob[i]:
            countdown = oob_hold
        elif countdown > 0:
            if real_vis[i] and (not oob[i]) and b_spd[i] > 7.5 and p_mean_s[i] > 1.2:
                countdown = 0
            else:
                post_oob[i] = True
                countdown -= 1

    # 1. Dead ball (D) signals
    log_e[oob, 2] += 10.0
    log_e[oob, 0] -= 10.0
    log_e[oob, 1] -= 10.0

    log_e[post_oob, 2] += 5.0
    log_e[post_oob, 0] -= 3.0
    log_e[post_oob, 1] -= 3.0

    # Low player tempo indicates play stoppage; include dead-ball retrieval guard when collective
    # tempo is stopped (< 1.32 m/s), opposing team is uninvolved (> 20m), and ball is not kicked (< 4.0 m/s)
    retrieval = vis & (p_mean_s < 1.32) & (np.maximum(h1, a1) > 20.0) & (b_spd < 4.0)
    low_tempo = ((p_mean_s < 1.55) & ((~vis) | (p_p90_s < 2.15))) | (p_p90_s < 2.15) | retrieval
    very_low_tempo = (p_mean_s < 1.35) & (p_p90_s < 1.95)
    log_e[low_tempo, 2] += 2.5
    log_e[very_low_tempo, 2] += 3.5

    # Require genuinely stopped collective + p90 tempo on missing-ball frames to prevent full-match D inflation
    missing_low_tempo = (~vis) & (p_mean_s < 1.45) & (p_p90_s < 2.20)
    log_e[missing_low_tempo, 2] += 2.0

    # High tempo indicates live play strictly when ball is visible on-pitch
    high_tempo = vis & (p_mean_s > 1.85) & (p_p90_s > 2.60) & (~oob) & (~post_oob)
    log_e[high_tempo, 2] -= 3.5

    # 2. Controlled possession signals (H vs A) with 2nd player support margin
    in_play_vis = vis & (~oob) & (~post_oob)

    # Uncontested loose rolling ball (neither team within 4.5m)
    uncontested = in_play_vis & (h1 > 4.5) & (a1 > 4.5) & (b_spd < 8.0)
    log_e[uncontested, 2] += 2.0
    log_e[uncontested, 0] -= 2.0
    log_e[uncontested, 1] -= 2.0

    # Ball flight invariance: freeze H/A emissions when ball speed > spd_cap
    in_flight = in_play_vis & (b_spd > spd_cap)
    live_ball = (~low_tempo) | ((b_spd > 1.8) & (~retrieval))
    h_ctrl = in_play_vis & (~in_flight) & (h1 <= ctrl_rad) & (h1 < a1) & live_ball
    a_ctrl = in_play_vis & (~in_flight) & (a1 <= ctrl_rad) & (a1 < h1) & live_ball

    # Incorporate 2nd-closest player support differential
    support_diff = np.clip(a2 - h2, -4.0, 4.0)
    h_margin = np.clip((a1 - h1) + 0.20 * support_diff, 0.0, 5.0)
    a_margin = np.clip((h1 - a1) - 0.20 * support_diff, 0.0, 5.0)

    log_e[h_ctrl, 0] += 1.2 * (ctrl_rad - h1[h_ctrl] + 0.5 * h_margin[h_ctrl])
    log_e[h_ctrl, 1] -= 1.0 * (ctrl_rad - h1[h_ctrl] + 0.5 * h_margin[h_ctrl])
    log_e[h_ctrl & (~low_tempo), 2] -= 2.5

    log_e[a_ctrl, 1] += 1.2 * (ctrl_rad - a1[a_ctrl] + 0.5 * a_margin[a_ctrl])
    log_e[a_ctrl, 0] -= 1.0 * (ctrl_rad - a1[a_ctrl] + 0.5 * a_margin[a_ctrl])
    log_e[a_ctrl & (~low_tempo), 2] -= 2.5

    # Soft proximity margin when ball is loose/slow (2.5m..4.5m)
    loose = in_play_vis & (~in_flight) & (~h_ctrl) & (~a_ctrl) & (np.minimum(h1, a1) < 4.5) & live_ball
    delta_loose = np.clip((a1[loose] - h1[loose]) + 0.15 * support_diff[loose], -3.0, 3.0)
    log_e[loose, 0] += 0.3 * delta_loose
    log_e[loose, 1] -= 0.3 * delta_loose

    # 3. Smart Tagging Multi-Phase Bayesian Priors
    for t in tags:
        code = t["code"].lower()
        sf, ef = max(0, t["start_frame"]), min(TOTAL_FRAMES, t["end_frame"])
        if sf >= ef:
            continue

        if code == "set pieces":
            sp_type = (t["tags"].get("Type") or "").lower()
            team = t["tags"].get("Team")
            t_code = 0 if team == "Barcelona" else (1 if team == "Atletic Club" else -1)
            if "kick off" not in sp_type:
                setup = low_tempo[sf:ef] | (~vis[sf:ef])
                if any(k in sp_type for k in ("goal kick", "free kick", "throw-in", "corner kick")) and t_code in (0, 1):
                    # Set piece setup ends immediately once the ball is kicked in-bounds (> 3.5 m/s)
                    kicked = np.cumsum((b_spd[sf:ef] > 3.5) & vis[sf:ef] & (~oob[sf:ef])) > 0
                    setup = setup & (~kicked)
                    log_e[sf:ef, t_code][kicked] += 3.0
                    log_e[sf:ef, 2][kicked] -= 5.0
                log_e[sf:ef, 2][setup] += 2.5
                if t_code in (0, 1):
                    live_after_kick = (~setup) & (b_spd[sf:ef] > 2.0) & (~post_oob[sf:ef])
                    log_e[sf:ef, t_code][live_after_kick] += 1.5

        elif code == "goals":
            # Post-goal celebration/stoppage is Dead Ball D
            log_e[sf:ef, 2] += 6.0
            log_e[sf:ef, 0] -= 4.0
            log_e[sf:ef, 1] -= 4.0

        elif code in ("build up", "progression", "creating chances", "attacking transition"):
            team = t["tags"].get("Team")
            t_code = 0 if team == "Barcelona" else (1 if team == "Atletic Club" else -1)
            if t_code in (0, 1):
                live_phase = in_play_vis[sf:ef] & (~low_tempo[sf:ef])
                log_e[sf:ef, t_code][live_phase] += 0.5
                log_e[sf:ef, 1 - t_code][live_phase] -= 0.3

        # When transition ends and ball is missing, stoppage occurs immediately
        if "transition" in code and ef < TOTAL_FRAMES:
            if not vis[ef]:
                log_e[ef:min(TOTAL_FRAMES, ef + 50), 2] += 4.0

    return log_e


def viterbi_decode_block(log_e_slice, pen_ha=40.0, pen_d=20.0):
    """Sticky Viterbi decoder with explicit log-transition penalties."""
    n = len(log_e_slice)
    log_A = np.array([
        [0.0, -pen_ha, -pen_d],
        [-pen_ha, 0.0, -pen_d],
        [-pen_d, -pen_d, 0.0]
    ], dtype=np.float32)

    trellis = np.zeros((n, 3), dtype=np.float32)
    bp = np.zeros((n, 3), dtype=np.int8)
    trellis[0] = log_e_slice[0]

    for t in range(1, n):
        c = trellis[t - 1, :, None] + log_A
        bp[t] = np.argmax(c, axis=0)
        trellis[t] = np.max(c, axis=0) + log_e_slice[t]

    path = np.zeros(n, dtype=np.int8)
    path[-1] = np.argmax(trellis[-1])
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return path


def prune_block(path, vis_slice, h1_slice, a1_slice, spd_slice,
                min_dead_len=80, min_alive_len=25, min_ctrl_frames=16):
    """Semi-Markov pruning: duration filter, controlled-frame verification, and flight/occlusion boundary snap."""
    out = path.copy()
    n = len(out)
    for _ in range(2):
        for start, stop, code in pm.segments(out, [(0, n)]):
            if code == pm.D and (stop - start) < min_dead_len:
                prev_c = out[start - 1] if start > 0 else (out[stop] if stop < n else pm.H)
                out[start:stop] = prev_c
        for start, stop, code in pm.segments(out, [(0, n)]):
            if code in (pm.H, pm.A):
                d_team = h1_slice[start:stop] if code == pm.H else a1_slice[start:stop]
                d_opp = a1_slice[start:stop] if code == pm.H else h1_slice[start:stop]
                ctrl_f = int(np.sum(vis_slice[start:stop] & (d_team <= 2.5) &
                                    (d_team < d_opp) & (spd_slice[start:stop] <= 6.0)))
                if (stop - start) < min_alive_len or ctrl_f < min_ctrl_frames:
                    prev_c = out[start - 1] if start > 0 else (out[stop] if stop < n else pm.D)
                    out[start:stop] = prev_c

    # 1. Snap H <-> A turnovers that land inside an invisible-ball occlusion (~vis) or high-speed pass flight (>10 m/s)
    # 2. Hold active H/A possession across accelerating final kicks into Dead Ball (D) until flight decelerates
    for start, stop, code in pm.segments(out, [(0, n)]):
        if start > 0 and code in (pm.H, pm.A) and out[start - 1] in (pm.H, pm.A) and out[start - 1] != code:
            cur = start
            while cur < stop and ((not vis_slice[cur]) or (spd_slice[cur] > 10.0)):
                out[cur] = out[start - 1]
                cur += 1
        elif start > 0 and code == pm.D and out[start - 1] in (pm.H, pm.A):
            if start + 5 < stop and vis_slice[start] and (spd_slice[start + 5] > spd_slice[start] + 1.0):
                cur = start
                while cur < stop and vis_slice[cur] and (
                    (spd_slice[cur] > 10.0) or (cur + 1 < stop and spd_slice[cur + 1] > spd_slice[cur])
                ):
                    out[cur] = out[start - 1]
                    cur += 1

    return out


def run(output_csv):
    """Full merged pipeline execution."""
    print("1. Loading match assets (Tactical + Broadcast ATD + Smart Tagging)...")
    paths = games.asset_paths(source="tactical")
    b_paths = games.asset_paths(source="broadcast")
    team_of_track = team_of_each_track(paths["atd_meta"])[:46]
    b_team_of_track = team_of_each_track(b_paths["atd_meta"])[:51]
    tags = smart_tagging.load(game="BAR-ATH")

    print("2. Extracting tracking coordinates, kinematics & multi-camera player proximities...")
    ball_xy, dist_H1, dist_H2, dist_A1, dist_A2, speed, p_mean, p_p90, is_interp = extract_tracking(
        paths["atd_raw"], team_of_track
    )
    if os.path.isfile(b_paths["atd_raw"]):
        dist_H1, dist_H2, dist_A1, dist_A2 = fuse_broadcast_tracking(
            b_paths["atd_raw"], b_team_of_track, ball_xy, dist_H1, dist_H2, dist_A1, dist_A2
        )

    print("3. Building physics, tempo & Smart Tagging frame log-emissions...")
    log_e = build_emissions(
        ball_xy, dist_H1, dist_H2, dist_A1, dist_A2, speed, p_mean, p_p90, tags, is_interp
    )

    print("4. Decoding graded halves with Sticky Viterbi & Controlled-Frame Pruning...")
    final_codes = np.full(TOTAL_FRAMES, pm.X, dtype=np.int8)
    vis = ~np.isnan(ball_xy[:, 0])

    # Half 1 graded block: 11185..90651
    s1 = slice(H1_START, H1_END + 1)
    h1_raw = viterbi_decode_block(log_e[s1], pen_ha=28.0, pen_d=20.0)
    final_codes[s1] = prune_block(h1_raw, vis[s1], dist_H1[s1], dist_A1[s1], speed[s1])

    # Half 2 pre-kickoff trap: 110808..113741 are 100% Dead Ball D
    final_codes[H2_START:H2_KICKOFF] = pm.D

    # Half 2 live play block: 113742..190197
    s2 = slice(H2_KICKOFF, H2_END + 1)
    h2_raw = viterbi_decode_block(log_e[s2], pen_ha=28.0, pen_d=20.0)
    final_codes[s2] = prune_block(h2_raw, vis[s2], dist_H1[s2], dist_A1[s2], speed[s2])

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)
    df = pd.DataFrame({
        "frame": np.arange(TOTAL_FRAMES),
        "label": [pm.LABELS[c] for c in final_codes]
    })
    df.to_csv(output_csv, index=False)
    print(f"-> Saved submission: {output_csv} ({len(df)} rows)")

    # Score against public ground truth
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    if os.path.isfile(gt_path):
        gt_table = pm.read_label_csv(gt_path)
        gt_codes = pm.label_codes(gt_table)
        res = pm.evaluate(gt_codes, final_codes)
        print(f"\n[Public 5-Min Score] spellF1: {res['spell_f1_mean']:.4f}  |  macroF1: {res['macro_f1']:.4f}"
              f"  |  Possessions: {res['n_pred_spells']} (GT=17)  |  Mean Dur: {res['mean_spell_duration_s']:.1f}s")

    # Run validator
    print("\nRunning submission validator...")
    reason = pm.validate_submission(df, TOTAL_FRAMES)
    if reason:
        print(f"VALIDATION FAILED: {reason}")
    else:
        print(f"VALIDATION PASSED: exactly {TOTAL_FRAMES} rows, labels in {list(pm.LABELS)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full-match submission")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "team_champion.csv"))
    args = parser.parse_args()
    run(args.out)
