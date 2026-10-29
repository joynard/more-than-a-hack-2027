"""More than a Hack 2027: Ball Possession (BAR-ATH) End-to-End Pipeline
Ponytail architecture: single clean file, stdlib + numpy/pandas, zero bloat.
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


def extract_tracking(atd_raw_path, team_of_track):
    """Stream 199,319 frames of raw ATD. Returns ball_xy, dist_H, dist_A, ball_speed, p_mean, p_p90."""
    ball_xy = np.full((TOTAL_FRAMES, 2), np.nan, dtype=np.float32)
    dist_H = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    dist_A = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    p_mean = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    p_p90 = np.zeros(TOTAL_FRAMES, dtype=np.float32)

    with open(atd_raw_path) as handle:
        for idx, line in enumerate(handle):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            bx, by = np.nan, np.nan
            if parts[2][0] != MISSING:
                ball = parts[2]
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx = float(ball[:c1]) * PITCH_LEN
                by = float(ball[c1 + 1:c2]) * PITCH_WID
                ball_xy[idx] = [bx, by]

            min_h, min_a = 1e5, 1e5
            spds = []
            has_ball = not np.isnan(bx)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                vals = group.split(",")
                s_val = float(vals[2])
                if not np.isnan(s_val):
                    spds.append(s_val)
                if has_ball:
                    px = float(vals[0]) * PITCH_LEN
                    py = float(vals[1]) * PITCH_WID
                    d = np.hypot(px - bx, py - by)
                    if team_of_track[slot] == HOME:
                        if d < min_h:
                            min_h = d
                    else:
                        if d < min_a:
                            min_a = d

            if spds:
                spds.sort()
                n_s = len(spds)
                p_mean[idx] = sum(spds) / n_s
                p_p90[idx] = spds[min(n_s - 1, int(0.9 * n_s))]
            if min_h < 1e4:
                dist_H[idx] = min_h
            if min_a < 1e4:
                dist_A[idx] = min_a

    # Ball speed (2-frame central difference)
    speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    valid = ~np.isnan(ball_xy[:-2, 0]) & ~np.isnan(ball_xy[2:, 0])
    dx = ball_xy[2:, 0][valid] - ball_xy[:-2, 0][valid]
    dy = ball_xy[2:, 1][valid] - ball_xy[:-2, 1][valid]
    speed[1:-1][valid] = np.hypot(dx, dy) / (2.0 / 25.0)

    return ball_xy, dist_H, dist_A, speed, p_mean, p_p90


def build_emissions(ball_xy, dist_H, dist_A, b_spd, p_mean, p_p90, tags,
                    ctrl_rad=2.5, spd_cap=10.0, oob_hold=230):
    """Physics & tempo log-emissions [N, 3] for states (0=H, 1=A, 2=D)."""
    log_e = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)

    # Smooth collective player tempo over 1-second (25-frame) window
    kernel = np.ones(25, dtype=np.float32) / 25.0
    p_mean_s = np.convolve(p_mean, kernel, mode="same")
    p_p90_s = np.convolve(p_p90, kernel, mode="same")

    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    vis = ~np.isnan(bx)
    oob = vis & ((bx < 0) | (bx > PITCH_LEN) | (by < 0) | (by > PITCH_WID))

    # Post-OOB hysteresis: once ball leaves pitch, play remains Dead (D) until restart kick
    post_oob = np.zeros(TOTAL_FRAMES, dtype=bool)
    countdown = 0
    for i in range(TOTAL_FRAMES):
        if oob[i]:
            countdown = oob_hold
        elif countdown > 0:
            if vis[i] and (not oob[i]) and b_spd[i] > 7.5 and p_mean_s[i] > 1.2:
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

    # Guard visible live play when top-10% player tempo (p_p90_s) is still active (>= 2.15 m/s)
    low_tempo = ((p_mean_s < 1.55) & ((~vis) | (p_p90_s < 2.15))) | (p_p90_s < 2.15)
    very_low_tempo = (p_mean_s < 1.35) & (p_p90_s < 1.95)
    log_e[low_tempo, 2] += 2.5
    log_e[very_low_tempo, 2] += 3.5

    missing_low_tempo = (~vis) & (p_mean_s < 1.75)
    log_e[missing_low_tempo, 2] += 4.0

    high_tempo = (p_mean_s > 1.85) & (p_p90_s > 2.60) & (~oob) & (~post_oob)
    log_e[high_tempo, 2] -= 3.5

    # 2. Controlled possession signals (H vs A)
    in_play_vis = vis & (~oob) & (~post_oob)
    h1 = np.nan_to_num(dist_H, nan=99.0)
    a1 = np.nan_to_num(dist_A, nan=99.0)

    # Ball flight invariance: freeze H/A emissions when ball speed > spd_cap
    in_flight = in_play_vis & (b_spd > spd_cap)
    live_ball = (~low_tempo) | (b_spd > 1.8)
    h_ctrl = in_play_vis & (~in_flight) & (h1 <= ctrl_rad) & (h1 < a1) & live_ball
    a_ctrl = in_play_vis & (~in_flight) & (a1 <= ctrl_rad) & (a1 < h1) & live_ball

    h_margin = np.clip(a1 - h1, 0.0, 5.0)
    a_margin = np.clip(h1 - a1, 0.0, 5.0)

    log_e[h_ctrl, 0] += 1.2 * (ctrl_rad - h1[h_ctrl] + 0.5 * h_margin[h_ctrl])
    log_e[h_ctrl, 1] -= 1.0 * (ctrl_rad - h1[h_ctrl] + 0.5 * h_margin[h_ctrl])
    log_e[h_ctrl & (~low_tempo), 2] -= 2.5

    log_e[a_ctrl, 1] += 1.2 * (ctrl_rad - a1[a_ctrl] + 0.5 * a_margin[a_ctrl])
    log_e[a_ctrl, 0] -= 1.0 * (ctrl_rad - a1[a_ctrl] + 0.5 * a_margin[a_ctrl])
    log_e[a_ctrl & (~low_tempo), 2] -= 2.5

    # Soft proximity margin when ball is loose/slow (2.5m..5.0m)
    loose = in_play_vis & (~in_flight) & (~h_ctrl) & (~a_ctrl) & (np.minimum(h1, a1) < 5.0) & live_ball
    delta_loose = np.clip(a1[loose] - h1[loose], -3.0, 3.0)
    log_e[loose, 0] += 0.3 * delta_loose
    log_e[loose, 1] -= 0.3 * delta_loose

    # 3. Smart Tagging Set Piece setup vs post-kick priors
    for t in tags:
        if t["code"].lower() == "set pieces":
            sf, ef = max(0, t["start_frame"]), min(TOTAL_FRAMES, t["end_frame"])
            if sf >= ef:
                continue
            sp_type = (t["tags"].get("Type") or "").lower()
            team = t["tags"].get("Team")
            t_code = 0 if team == "Barcelona" else (1 if team == "Atletic Club" else -1)
            if "kick off" not in sp_type:
                setup = low_tempo[sf:ef] | (~vis[sf:ef])
                if "goal kick" in sp_type and t_code in (0, 1):
                    # Goal kick setup ends immediately once the ball is kicked in-bounds (> 3.5 m/s)
                    kicked = np.cumsum((b_spd[sf:ef] > 3.5) & vis[sf:ef] & (~oob[sf:ef])) > 0
                    setup = setup & (~kicked)
                    log_e[sf:ef, t_code][kicked] += 3.0
                    log_e[sf:ef, 2][kicked] -= 5.0
                log_e[sf:ef, 2][setup] += 2.5
                if t_code in (0, 1):
                    live_after_kick = (~setup) & (b_spd[sf:ef] > 2.0) & (~post_oob[sf:ef])
                    log_e[sf:ef, t_code][live_after_kick] += 1.5

    return log_e


def viterbi_decode_block(log_e_slice, pen_ha=45.0, pen_d=20.0):
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
                min_dead_len=80, min_alive_len=25, min_ctrl_frames=12):
    """Semi-Markov pruning: duration filter, controlled-frame verification, and occlusion boundary snap."""
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

    # Snap H <-> A turnovers that land inside an invisible-ball occlusion (~vis) to the first visible frame
    for start, stop, code in pm.segments(out, [(0, n)]):
        if start > 0 and code in (pm.H, pm.A) and out[start - 1] in (pm.H, pm.A) and out[start - 1] != code:
            cur = start
            while cur < stop and (not vis_slice[cur]):
                out[cur] = out[start - 1]
                cur += 1

    return out


def run(output_csv):
    """Full pipeline run."""
    print("1. Loading match assets...")
    paths = games.asset_paths(source="tactical")
    team_of_track = team_of_each_track(paths["atd_meta"])
    tags = smart_tagging.load(game="BAR-ATH")

    print("2. Extracting tracking coordinates, kinematics & collective player tempo...")
    ball_xy, dist_H, dist_A, speed, p_mean, p_p90 = extract_tracking(paths["atd_raw"], team_of_track)

    print("3. Building physics & tempo frame log-emissions...")
    log_e = build_emissions(ball_xy, dist_H, dist_A, speed, p_mean, p_p90, tags)

    print("4. Decoding graded halves with Sticky Viterbi & Controlled-Frame Pruning...")
    final_codes = np.full(TOTAL_FRAMES, pm.X, dtype=np.int8)
    vis = ~np.isnan(ball_xy[:, 0])
    h1 = np.nan_to_num(dist_H, nan=99.0)
    a1 = np.nan_to_num(dist_A, nan=99.0)

    # Half 1 graded block: 11185..90651
    s1 = slice(H1_START, H1_END + 1)
    h1_raw = viterbi_decode_block(log_e[s1], pen_ha=45.0, pen_d=20.0)
    final_codes[s1] = prune_block(h1_raw, vis[s1], h1[s1], a1[s1], speed[s1])

    # Half 2 pre-kickoff trap: 110808..113741 are 100% Dead Ball D
    final_codes[H2_START:H2_KICKOFF] = pm.D

    # Half 2 live play block: 113742..190197
    s2 = slice(H2_KICKOFF, H2_END + 1)
    h2_raw = viterbi_decode_block(log_e[s2], pen_ha=45.0, pen_d=20.0)
    final_codes[s2] = prune_block(h2_raw, vis[s2], h1[s2], a1[s2], speed[s2])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full-match submission")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "team_ponytail.csv"))
    args = parser.parse_args()
    run(args.out)
