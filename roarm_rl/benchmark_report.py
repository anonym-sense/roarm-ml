"""Turn the logs written by roarm_rl.benchmark into the numbers and figures in docs/paper.

    python -m roarm_rl.benchmark_report

Reads docs/paper/data/*.json, writes docs/paper/figures/*.png, docs/paper/results.json
(every number quoted in the paper) and docs/paper/tables.md (the same as tables).
Experiments whose log is missing are skipped.
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roarm_rl.benchmark import COUNT, DATA_DIR, JOINTS, Kin  # noqa: E402

PAPER_DIR = os.path.dirname(DATA_DIR)
FIG_DIR = os.path.join(PAPER_DIR, "figures")

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
JOINT_COLOR = dict(zip(JOINTS, (BLUE, ORANGE, AQUA, YELLOW)))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 1.6, "figure.dpi": 140,
})


def load(name):
    path = os.path.join(DATA_DIR, name + ".json")
    if not os.path.exists(path):
        print(f"(no {name}.json, skipped)")
        return None
    with open(path) as f:
        return json.load(f)


def savefig(fig, name):
    os.makedirs(os.path.dirname(os.path.join(FIG_DIR, name)), exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)


def arrays(run):
    return np.array(run["t"]), np.array(run["q"]), np.array(run["ref"])


def settled(run, window=0.4):
    """Mean and spread of the encoder readings over the last `window` seconds of a run."""
    t, q, _ = arrays(run)
    tail = q[t >= t[-1] - window]
    return tail.mean(axis=0), tail.std(axis=0), np.array(run["load"])[t >= t[-1] - window].mean(axis=0)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def best_lag(t, measured, ref_t, ref, lo=-0.2, hi=0.5, step=0.005):
    """Delay (s) of `measured` behind `ref` that makes them agree best, and the error left over."""
    best = (float("inf"), 0.0)
    for lag in np.arange(lo, hi + 1e-9, step):
        shifted = np.column_stack([np.interp(t - lag, ref_t, ref[:, k]) for k in range(ref.shape[1])])
        best = min(best, (rms(measured - shifted), float(lag)))
    return best[1], best[0]


# --- static accuracy --------------------------------------------------------------

def report_static(data, out):
    rows = {}
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    for j, name in enumerate(JOINTS):
        runs = [r for r in data["runs"] if r["joint"] == name]
        pts = {"up": {}, "down": {}}
        noise, settle, loads = [], [], []
        for r in runs:
            mean, std, load_ = settled(r)
            pts[r["direction"]].setdefault(round(r["target"], 4), []).append(mean[j] - r["target"])
            noise.append(std[j])
            loads.append((r["target"], load_[j]))
            t, q, _ = arrays(r)
            moving = np.abs(q[:, j] - mean[j]) > 2 * COUNT
            ramp_end = r["seconds"]
            settle.append(max(0.0, (t[moving][-1] if moving.any() else 0.0) - ramp_end))
        errors = np.array([e for d in pts.values() for v in d.values() for e in v])
        common = sorted(set(pts["up"]) & set(pts["down"]))
        hysteresis = [np.mean(pts["up"][k]) - np.mean(pts["down"][k]) for k in common]
        repeat = [np.std(v, ddof=1) for d in pts.values() for v in d.values() if len(v) > 1]
        rows[name] = {
            "bias_mrad": 1e3 * float(errors.mean()), "rms_mrad": 1e3 * rms(errors),
            "max_mrad": 1e3 * float(np.abs(errors).max()),
            "hysteresis_mrad": 1e3 * float(np.mean(hysteresis)),
            "repeat_mrad": 1e3 * float(np.sqrt(np.mean(np.square(repeat)))),
            "noise_mrad": 1e3 * float(np.mean(noise)), "settle_s": float(np.median(settle)),
            "settle_max_s": float(np.max(settle)), "n": int(errors.size),
        }
        ax = axes[j]
        for direction, marker, color in (("up", "^", JOINT_COLOR[name]), ("down", "v", MUTED)):
            xs = sorted(pts[direction])
            ax.errorbar(xs, [1e3 * np.mean(pts[direction][x]) for x in xs],
                        yerr=[1e3 * (np.ptp(pts[direction][x]) / 2) for x in xs], marker=marker,
                        ms=5, color=color, lw=1.2, capsize=2, label=f"approached going {direction}")
        ax.axhline(0, color=INK, lw=0.6)
        ax.axhspan(-1e3 * COUNT, 1e3 * COUNT, color=GRID, alpha=0.6, lw=0)
        ax.set_title(name)
        ax.set_xlabel("target (rad)")
        if j == 0:
            ax.set_ylabel("settled error, measured − target (mrad)")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Where each joint settles, by target and direction of approach "
                 "(band = ±1 encoder count)", x=0.01, y=1.06, ha="left", fontsize=10)
    savefig(fig, "static_accuracy.png")
    out["static"] = rows


def report_poses(data, kin, out):
    by_pose = {}
    for r in data["runs"]:
        mean, _, _ = settled(r)
        by_pose.setdefault(r["pose"], []).append((np.array(r["target"]), mean))
    acc, rep, joint_err = [], [], []
    for visits in by_pose.values():
        tcps = []
        for target, mean in visits:
            tcps.append(np.array(kin.fk(mean)))
            acc.append(1e3 * math.dist(tcps[-1], kin.fk(target)))
            joint_err.append(mean - target)
        if len(tcps) > 1:
            rep.append(1e3 * math.dist(tcps[0], tcps[1]))
    joint_err = np.array(joint_err)
    out["poses"] = {
        "n_poses": len(by_pose), "n_visits": len(acc),
        "tcp_mean_mm": float(np.mean(acc)), "tcp_median_mm": float(np.median(acc)),
        "tcp_p95_mm": float(np.percentile(acc, 95)), "tcp_max_mm": float(np.max(acc)),
        "repeat_mean_mm": float(np.mean(rep)), "repeat_max_mm": float(np.max(rep)),
        "joint_rms_mrad": {n: 1e3 * rms(joint_err[:, j]) for j, n in enumerate(JOINTS)},
        "joint_bias_mrad": {n: 1e3 * float(joint_err[:, j].mean()) for j, n in enumerate(JOINTS)},
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.9))
    a.hist(acc, bins=14, color=BLUE, edgecolor=SURFACE)
    a.set_title("Hand position error at 24 whole-arm poses (2 visits each)")
    a.set_xlabel("distance from the commanded hand position (mm)")
    a.set_ylabel("visits")
    b.hist(rep, bins=12, color=AQUA, edgecolor=SURFACE)
    b.set_title("Distance between the two visits to the same pose")
    b.set_xlabel("mm")
    savefig(fig, "pose_accuracy.png")


# --- dynamics ---------------------------------------------------------------------

def step_metrics(run):
    j = JOINTS.index(run["joint"])
    t, q, _ = arrays(run)
    q = q[:, j]
    start, target = q[0], run["target"][j]
    span = target - start
    sign = np.sign(span)
    moved = np.abs(q - start) > 2 * COUNT
    dead = float(t[moved][0]) if moved.any() else float("nan")
    frac = (q - start) / span
    t10 = float(t[np.argmax(frac >= 0.1)])
    t90 = float(t[np.argmax(frac >= 0.9)]) if (frac >= 0.9).any() else float("nan")
    v = np.gradient(q, t)
    final = q[t >= t[-1] - 0.3].mean()
    outside = np.abs(q - final) > 0.01
    return {
        "dead_s": dead, "rise_s": t90 - t10, "peak_speed": float(np.abs(v).max()),
        "overshoot_mrad": 1e3 * float(max(0.0, (sign * (q - final)).max())),
        "settle_s": float(t[outside][-1]) if outside.any() else 0.0,
        "final_error_mrad": 1e3 * float(final - target),
    }


def report_step(data, out):
    table = []
    for r in data["runs"]:
        table.append({"joint": r["joint"], "size": r["size"], "profile": r["profile"],
                      "direction": r["direction"], **step_metrics(r)})
    summary = {}
    for name in JOINTS:
        for profile in ([1500, 60], [880, 60], [300, 10]):
            rows = [m for m in table if m["joint"] == name and m["profile"] == profile and m["size"] == 0.5]
            summary[f"{name} {profile[0]}/{profile[1]}"] = {
                k: float(np.mean([m[k] for m in rows]))
                for k in ("dead_s", "rise_s", "peak_speed", "overshoot_mrad", "settle_s", "final_error_mrad")}
    fast = [m for m in table if m["profile"] == [1500, 60]]
    out["step"] = {
        "by_profile": summary,
        "dead_s_mean": float(np.nanmean([m["dead_s"] for m in table])),
        "dead_s_min": float(np.nanmin([m["dead_s"] for m in table])),
        "dead_s_max": float(np.nanmax([m["dead_s"] for m in table])),
        "peak_speed_fast": {n: float(max(m["peak_speed"] for m in fast if m["joint"] == n)) for n in JOINTS},
        "overshoot_max_mrad": float(max(m["overshoot_mrad"] for m in table)),
        "small_step_error_mrad": {n: float(np.mean([abs(m["final_error_mrad"]) for m in fast
                                                    if m["joint"] == n and m["size"] == 0.05]))
                                  for n in JOINTS},
        "runs": table,
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    styles = {(1500, 60): (BLUE, "following: 1500 / 60"), (880, 60): (ORANGE, "planned: 880 / 60"),
              (300, 10): (AQUA, "catching up: 300 / 10")}
    for j, name in enumerate(JOINTS):
        ax = axes[j]
        for r in data["runs"]:
            if r["joint"] != name or r["size"] != 0.5 or r["direction"] != "up":
                continue
            t, q, _ = arrays(r)
            color, label = styles[tuple(r["profile"])]
            ax.plot(t, q[:, j] - r["start"][j], color=color, label=label)
        ax.axhline(0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlim(0, 2.2)
        ax.set_title(name)
        ax.set_xlabel("time after the command (s)")
        if j == 0:
            ax.set_ylabel("measured move (rad)")
            ax.legend(fontsize=7, title="servo speed / acceleration", title_fontsize=7)
    fig.suptitle("One 0.5 rad command per joint at the web app's three servo settings (dashed = target)",
                 x=0.01, y=1.06, ha="left", fontsize=10)
    savefig(fig, "step_response.png")


def sine_fit(run):
    j = JOINTS.index(run["joint"])
    t, q, ref = arrays(run)
    hz, amp = run["hz"], run["amplitude"]
    keep = (t >= 1.0 / hz) & (t <= run["seconds"])
    w = 2 * np.pi * hz * t[keep]
    basis = np.column_stack([np.sin(w), np.cos(w), np.ones(w.size)])
    (a, b, c), *_ = np.linalg.lstsq(basis, q[keep, j], rcond=None)
    fitted = basis @ np.array([a, b, c])
    phase = -math.atan2(b, a)  # radians the arm is behind
    if phase < -0.3:
        phase += 2 * math.pi
    v = np.gradient(q[:, j], t)
    return {"joint": run["joint"], "hz": hz, "amplitude": amp, "gain": float(math.hypot(a, b) / amp),
            "lag_s": float(phase / (2 * np.pi * hz)), "phase_deg": float(math.degrees(phase)),
            "offset_mrad": 1e3 * float(c - run["centre"][j]),
            "distortion": rms(q[keep, j] - fitted) / (amp / math.sqrt(2)),
            "rms_error_mrad": 1e3 * rms(q[keep, j] - ref[keep, j]),
            "peak_speed": float(np.abs(v[keep]).max()), "asked_speed": float(2 * np.pi * hz * amp),
            "commands_hz": float(len(run["sent"]) / run["seconds"])}


def report_sine(data, out):
    fits = [sine_fit(r) for r in data["runs"]]
    out["sine"] = {"fits": fits,
                   "commands_hz": float(np.mean([f["commands_hz"] for f in fits])),
                   "lag_low_s": {n: float(np.mean([f["lag_s"] for f in fits if f["joint"] == n
                                                   and f["hz"] <= 0.7 and f["amplitude"] == 0.15]))
                                 for n in JOINTS},
                   "speed_cap": {n: float(max(f["peak_speed"] for f in fits if f["joint"] == n))
                                 for n in JOINTS}}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    for name in JOINTS:
        for amp, ls, marker in ((0.15, "-", "o"), (0.4, (0, (4, 2)), "s")):
            rows = sorted((f for f in fits if f["joint"] == name and f["amplitude"] == amp),
                          key=lambda f: f["hz"])
            hz = [f["hz"] for f in rows]
            label = f"{name}, ±{amp} rad"
            axes[0].plot(hz, [f["gain"] for f in rows], ls=ls, marker=marker, ms=4,
                         color=JOINT_COLOR[name], label=label)
            axes[1].plot(hz, [1e3 * f["lag_s"] for f in rows], ls=ls, marker=marker, ms=4,
                         color=JOINT_COLOR[name])
            axes[2].plot([f["asked_speed"] for f in rows], [f["peak_speed"] for f in rows], ls="",
                         marker=marker, ms=5, color=JOINT_COLOR[name])
    axes[0].axhline(1, color=INK, lw=0.6)
    axes[0].set_title("Size of the arm's swing ÷ size asked for")
    axes[1].set_title("How far behind the arm runs (ms)")
    axes[2].set_title("Peak speed reached vs asked (rad/s)")
    top = max(f["asked_speed"] for f in fits)
    axes[2].plot([0, top], [0, top], color=INK, lw=0.6)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.4, 0.7, 1, 1.5, 2, 3], ["0.2", "0.4", "0.7", "1", "1.5", "2", "3"])
        ax.minorticks_off()
        ax.set_xlabel("sine frequency (Hz)")
    axes[2].set_xlabel("peak speed of the commanded sine (rad/s)")
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    savefig(fig, "sine_response.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.7))
    for ax, (hz, amp) in zip(axes, ((0.4, 0.15), (1.5, 0.15), (1.0, 0.4))):
        r = next(r for r in data["runs"] if r["joint"] == "shoulder" and r["hz"] == hz
                 and r["amplitude"] == amp)
        t, q, ref = arrays(r)
        fine = np.linspace(0, r["seconds"], 800)
        ax.plot(fine, amp * np.sin(2 * np.pi * hz * fine), color=MUTED, lw=1.2, label="commanded")
        ax.plot(t, q[:, 1] - r["centre"][1], color=ORANGE, label="measured")
        ax.set_xlim(0, min(r["seconds"], 4.0))
        ax.set_title(f"shoulder, {hz} Hz, ±{amp} rad")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("rad about the centre")
    axes[0].legend(fontsize=7, loc="upper right")
    savefig(fig, "sine_examples.png")


def report_stream(data, out):
    rows = []
    for r in data["runs"]:
        t, q, ref = arrays(r)
        keep = (t > 2.0) & (t < r["seconds"])
        fine = np.linspace(0, r["seconds"], 2000)
        fine_ref = np.array([[STREAM_REF(x)[k] for k in (0, 2)] for x in fine])
        lag, left = best_lag(t[keep], q[keep][:, [0, 2]], fine, fine_ref)
        v = np.gradient(q[:, [0, 2]], t, axis=0)
        v_ref = np.column_stack([np.interp(t - lag, fine, np.gradient(fine_ref[:, k], fine)) for k in (0, 1)])
        label = {"speed 300": "speed 300, acc 10 (catching up)",
                 "speed 880": "speed 880 (planned moves)"}.get(r["label"], r["label"])
        rows.append({"label": label, "commands_hz": len(r["sent"]) / r["seconds"],
                     "rms_mrad": 1e3 * rms(q[keep][:, [0, 2]] - ref[keep][:, [0, 2]]),
                     "lag_s": lag, "rms_at_lag_mrad": 1e3 * left,
                     "speed_ripple": rms((v - v_ref)[keep])})
    out["stream"] = rows
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    labels = [r["label"] for r in rows][::-1]
    for ax, key, title in zip(axes, ("rms_mrad", "rms_at_lag_mrad", "speed_ripple"),
                              ("Error against the reference (mrad RMS)",
                               "Same, with the delay removed (mrad RMS)",
                               "Speed ripple (rad/s RMS)")):
        values = [r[key] for r in rows][::-1]
        ax.barh(labels, values, color=[ORANGE if "web app" in s else BLUE for s in labels], height=0.62)
        for y, value in enumerate(values):
            ax.text(value, y, f" {value:.2f}" if key == "speed_ripple" else f" {value:.0f}",
                    va="center", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    savefig(fig, "stream_variants.png")


def report_limits(data, out):
    rows = []
    for r in data["runs"]:
        f = sine_fit(r)
        rows.append({"hz": f["hz"], "speed": r["profile"][0], "acc": r["profile"][1], "gain": f["gain"],
                     "lag_s": f["lag_s"], "distortion": f["distortion"], "peak_speed": f["peak_speed"],
                     "rms_error_mrad": f["rms_error_mrad"]})
    out["limits"] = rows
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    for ax, hz in zip(axes, (1.5, 2.0)):
        for profile, color in (((1500, 60), ORANGE), ((1500, 254), BLUE), ((4000, 0), AQUA)):
            r = next(r for r in data["runs"] if r["hz"] == hz and tuple(r["profile"]) == profile)
            t, q, _ = arrays(r)
            ax.plot(t, q[:, 0] - r["centre"][0], color=color, label=f"speed {profile[0]}, acc {profile[1]}")
        fine = np.linspace(0, r["seconds"], 800)
        ax.plot(fine, 0.15 * np.sin(2 * np.pi * hz * fine), color=MUTED, lw=1.1, label="commanded", zorder=1)
        ax.set_xlim(0, 2.7)
        ax.set_title(f"base, {hz} Hz, ±0.15 rad")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("rad about the centre")
    axes[1].legend(fontsize=7, loc="upper right")
    savefig(fig, "acceleration_limit.png")


def report_hold(data, out):
    rows = []
    for r in data["runs"]:
        t, q, _ = arrays(r)
        load_ = np.array(r["load"])
        keep = t > 1.0
        grid = np.arange(t[keep][0], t[-1], 0.025)
        row = {"shoulder": r["target"][1], "elbow": r["target"][2],
               "load": [float(v) for v in load_[keep].mean(axis=0)]}
        for j, name in enumerate(JOINTS):
            x = q[keep, j] - q[keep, j].mean()
            row[name + "_pp_mrad"] = 1e3 * float(np.percentile(x, 97.5) - np.percentile(x, 2.5))
            spectrum = np.abs(np.fft.rfft(np.interp(grid, t[keep], x) * np.hanning(grid.size)))
            freqs = np.fft.rfftfreq(grid.size, 0.025)
            row[name + "_hz"] = float(freqs[1:][np.argmax(spectrum[1:])])
        row["error_mrad"] = [1e3 * float(v) for v in q[keep].mean(axis=0) - np.array(r["target"])]
        rows.append(row)
    hunting = [r for r in rows if r["shoulder_pp_mrad"] > 4 * 1e3 * COUNT]
    out["hold"] = {
        "rows": rows, "n": len(rows), "hunting": len(hunting),
        "hunting_shoulder_range": [min(r["shoulder"] for r in hunting), max(r["shoulder"] for r in hunting)]
        if hunting else None,
        "hunting_pp_max_mrad": max((r["shoulder_pp_mrad"] for r in hunting), default=0.0),
        "hunting_hz": float(np.median([r["shoulder_hz"] for r in hunting])) if hunting else None,
        "other_joints_pp_max_mrad": {n: max(r[n + "_pp_mrad"] for r in rows) for n in JOINTS if n != "shoulder"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for elbow, color in zip(sorted({r["elbow"] for r in rows}), (BLUE, ORANGE, AQUA)):
        sub = sorted((r for r in rows if r["elbow"] == elbow), key=lambda r: r["shoulder"])
        xs = [r["shoulder"] for r in sub]
        axes[0].plot(xs, [r["shoulder_pp_mrad"] for r in sub], marker="o", ms=4, color=color,
                     label=f"elbow at {elbow:.2f} rad")
        axes[1].plot(xs, [r["load"][1] for r in sub], marker="o", ms=4, color=color)
    axes[0].axhline(2e3 * COUNT, color=INK, lw=0.6, ls=(0, (4, 3)))
    axes[0].set_title("Shoulder movement at rest (mrad peak to peak)")
    axes[1].axhline(0, color=INK, lw=0.6)
    axes[1].set_title("Load the shoulder servo reports")
    for ax in axes[:2]:
        ax.set_xlabel("shoulder angle (rad; negative leans back)")
    axes[0].legend(fontsize=7)
    worst = max(data["runs"], key=lambda r: np.ptp(np.array(r["q"])[np.array(r["t"]) > 1.0, 1]))
    t, q, _ = arrays(worst)
    axes[2].plot(t, 1e3 * (q[:, 1] - worst["target"][1]), color=ORANGE)
    axes[2].set_xlim(1, 5)
    axes[2].set_title(f"Worst case (shoulder {worst['target'][1]:.2f} rad)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("measured − target (mrad)")
    savefig(fig, "hold_stability.png")


def STREAM_REF(t):
    return [0.3 * math.sin(2 * math.pi * 0.5 * t), 0.0,
            1.45 + 0.25 * math.sin(2 * math.pi * 0.5 * t + math.pi / 2) - 0.25, 0.6]


# --- inverse kinematics -----------------------------------------------------------

def report_iksim(data, out):
    once = 1e3 * np.array([r["err_once"] for r in data["reachable"]])
    four = 1e3 * np.array([r["err_four"] for r in data["reachable"]])
    box = 1e3 * np.array([r["err_four"] for r in data["box"]])
    ms = np.array([r["ms"] for r in data["reachable"]])

    def stats(x):
        return {"median_mm": float(np.median(x)), "p95_mm": float(np.percentile(x, 95)),
                "max_mm": float(x.max()), "over_1mm": float(np.mean(x > 1.0)),
                "over_10mm": float(np.mean(x > 10.0))}
    out["iksim"] = {"n": int(once.size), "once": stats(once), "four": stats(four),
                    "box": {**stats(box), "n": int(box.size), "within_5mm": float(np.mean(box < 5.0))},
                    "solve_ms": float(np.median(ms))}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3))
    bins = np.logspace(-4, 2.7, 48)
    a.hist(np.clip(once, 1e-4, None), bins=bins, color=ORANGE, alpha=0.85,
           label="one solve from the current pose (dragging the hand)")
    a.hist(np.clip(four, 1e-4, None), bins=bins, color=BLUE, alpha=0.75,
           label="four solves from home (hand-over)")
    a.set_xscale("log")
    a.set_xlabel("distance between the target and where the solution puts the hand (mm)")
    a.set_ylabel("targets")
    a.set_title(f"IK residual on {once.size} reachable targets")
    a.legend(fontsize=7)
    xyz = np.array([r["xyz"] for r in data["box"]])
    sc = b.scatter(np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2], c=np.clip(box, 0, 100), s=6, cmap="Blues",
                   vmin=0, vmax=100)
    fig.colorbar(sc, ax=b, label="residual (mm, capped at 100)")
    b.set_xlabel("distance from the base axis (m)")
    b.set_ylabel("height (m)")
    b.set_title("Any target in a box around the arm (pale = reachable in the safe range)")
    savefig(fig, "ik_simulation.png")


def report_ik(data, kin, out):
    rows = []
    for r in data["runs"]:
        mean, _, _ = settled(r)
        tcp = kin.fk(mean)
        rows.append({"ik_mm": 1e3 * r["ik_error"], "reached_mm": 1e3 * math.dist(tcp, r["hand"]),
                     "servo_mm": 1e3 * math.dist(tcp, kin.fk(r["goal"])),
                     "joint_mrad": [1e3 * (a - b) for a, b in zip(mean, r["goal"])]})
    reached = np.array([r["reached_mm"] for r in rows])
    out["ik"] = {"n": len(rows), "ik_median_mm": float(np.median([r["ik_mm"] for r in rows])),
                 "ik_max_mm": float(max(r["ik_mm"] for r in rows)),
                 "reached_mean_mm": float(reached.mean()), "reached_median_mm": float(np.median(reached)),
                 "reached_max_mm": float(reached.max()),
                 "servo_mean_mm": float(np.mean([r["servo_mm"] for r in rows]))}
    order = np.argsort(reached)
    fig, ax = plt.subplots(figsize=(9, 2.9))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [rows[i]["ik_mm"] for i in order], width=0.38, color=MUTED, label="IK residual (simulator)")
    ax.bar(x + 0.2, reached[order], width=0.38, color=BLUE, label="target to where the encoders say the hand is")
    ax.set_xticks([])
    ax.set_xlabel("24 hand targets, sorted by error")
    ax.set_ylabel("mm")
    ax.set_title("Reaching a point: how much of the miss is IK, how much is the servos")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    savefig(fig, "ik_hardware.png")


def report_firmware_model(logs, kin, out):
    """The arm's own idea of its hand position (its feedback x, y, z) against the URDF's."""
    pairs = []
    for data in logs:
        for r in data["runs"]:
            mean, _, _ = settled(r)
            t = np.array(r["t"])
            xyz = np.array(r["xyz"])[t >= t[-1] - 0.4].mean(axis=0) / 1e3
            pairs.append((xyz, kin.fk(mean)))
    fw, urdf = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    offset = (urdf - fw).mean(axis=0)
    left = np.linalg.norm(urdf - fw - offset, axis=1)
    out["firmware_model"] = {"n": len(pairs), "offset_mm": [1e3 * float(v) for v in offset],
                             "residual_mean_mm": 1e3 * float(left.mean()),
                             "residual_max_mm": 1e3 * float(left.max())}


# --- animations -------------------------------------------------------------------

def path_distance(points, path):
    """Distance from each point to the nearest place on a polyline (same units)."""
    points, path = np.asarray(points), np.asarray(path)
    a, b = path[:-1], path[1:]
    ab = b - a
    out = []
    for p in points:
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        out.append(np.min(np.linalg.norm(a + u[:, None] * ab - p, axis=1)))
    return np.array(out)


def analyse_animation(run, kin):
    t, q, _ = arrays(run)
    pic = np.array(run["picture"])
    pt, pq = pic[:, 0], pic[:, 1:]
    _, first = np.unique(pt, return_index=True)
    pt, pq = pt[first], pq[first]
    moving = np.abs(pq - pq[0]).max(axis=1) > 1e-4
    end = pt[moving][-1] if moving.any() else pt[-1]
    keep = (t >= 0) & (t <= end + 0.3)
    ref = np.column_stack([np.interp(t, pt, pq[:, k]) for k in range(4)])
    err = (q - ref)[keep]
    tcp_meas = np.array([kin.fk(x) for x in q])
    tcp_pic = np.array([kin.fk(x) for x in ref])
    tcp_err = 1e3 * np.linalg.norm(tcp_meas - tcp_pic, axis=1)
    lag, _ = best_lag(t[keep], q[keep][:, :3], pt, pq[:, :3])
    ref_lag = np.column_stack([np.interp(t - lag, pt, pq[:, k]) for k in range(4)])
    tcp_lag = 1e3 * np.linalg.norm(tcp_meas - np.array([kin.fk(x) for x in ref_lag]), axis=1)
    reach = []
    for k in range(4):
        asked = np.ptp(pq[:, k])
        if asked > 0.15:
            reach.append(np.ptp(q[keep][:, k]) / asked)
    v = np.gradient(q[:, :3], t, axis=0)
    v_ref = np.column_stack([np.interp(t - lag, pt, np.gradient(pq[:, k], pt)) for k in range(3)])
    rest = t > end + 0.5
    if not rest.any():
        rest = t >= t[-3]
    m = {
        "name": run["name"], "kind": run["kind"], "repeat": run["repeat"],
        "designed_s": sum(d for d, _ in run["designed"][1:]), "played_s": float(end),
        "joint_rms_mrad": [1e3 * rms(err[:, k]) for k in range(4)],
        "joint_max_mrad": [1e3 * float(np.abs(err[:, k]).max()) for k in range(4)],
        "arm_rms_mrad": 1e3 * rms(err[:, :3]),
        "tcp_rms_mm": rms(tcp_err[keep]), "tcp_max_mm": float(tcp_err[keep].max()),
        "lag_s": lag, "tcp_rms_at_lag_mm": rms(tcp_lag[keep]),
        "reach": float(min(reach)) if reach else 1.0,
        "speed_ripple": rms((v - v_ref)[keep]),
        "peak_speed": float(np.abs(np.gradient(pq[:, :3], pt, axis=0)).max()),
        "end_error_mm": float(tcp_err[rest].mean()),
        "commands_hz": len(run["sent"]) / max(end, 1e-6),
        "travel_mm": 1e3 * float(np.linalg.norm(tcp_pic[keep] - tcp_pic[keep][0], axis=1).max()),
    }
    m["relative"] = m["tcp_rms_mm"] / m["travel_mm"]
    series = {"t": t, "q": q, "ref": ref, "tcp_meas": tcp_meas, "tcp_pic": tcp_pic, "tcp_err": tcp_err,
              "keep": keep, "end": end, "pt": pt, "pq": pq}
    if run.get("path"):
        played = run["played"]
        start = played[1][0] + 0.05
        stop = start + sum(d for d, _ in played[2:-1])
        on_pic = (t >= start) & (t <= stop)
        on_meas = (t >= start + lag) & (t <= stop + lag)
        path = np.array(run["path"])
        m["path_error_picture_mm"] = 1e3 * rms(path_distance(tcp_pic[on_pic], path))
        d = 1e3 * path_distance(tcp_meas[on_meas], path)
        m["path_error_mm"], m["path_error_max_mm"] = rms(d), float(d.max())
        m["path_size_mm"] = 1e3 * float(np.ptp(path, axis=0).max())
        series["on_meas"], series["on_pic"] = on_meas, on_pic
    return m, series


def score(m):
    """0 to 10 for one animation; the same rubric is spelled out in the paper."""
    lost = (min(4.0, m["tcp_rms_mm"] / 5.0) + min(2.0, abs(m["lag_s"]) / 0.05)
            + min(2.0, max(0.0, 1.0 - m["reach"]) / 0.05) + min(1.0, m["repeat_mm"] / 3.0)
            + min(1.0, m["end_error_mm"] / 5.0))
    return round(10.0 - lost, 1)


def plot_animation(name, runs, kin):
    """Joint traces, hand error and (for shapes) the traced path for one animation."""
    (m, s), has_path = runs[0], "on_meas" in runs[0][1]
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(2, 4 if has_path else 3, width_ratios=[1, 1, 1, 1.25][:4 if has_path else 3])
    for k, joint in enumerate(JOINTS):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        ax.plot(s["pt"], s["pq"][:, k], color=MUTED, lw=1.2, label="picture (simulated)")
        ax.plot(s["t"], s["q"][:, k], color=JOINT_COLOR[joint], label="measured")
        ax.set_xlim(0, s["end"] + 0.8)
        ax.set_title(f"{joint} (rad)")
        if k >= 2:
            ax.set_xlabel("time (s)")
        if k == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(grid[:, 2])
    for (_, r), color, label in zip(runs, (BLUE, AQUA), ("run 1", "run 2")):
        ax.plot(r["t"][r["keep"]], r["tcp_err"][r["keep"]], color=color, lw=1.2, label=label)
    ax.set_title("hand: measured − picture (mm)")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7)
    if has_path:
        ax = fig.add_subplot(grid[:, 3])
        path = np.array(runs[0][0]["path_xyz"])
        spread = np.ptp(path, axis=0)
        i, j = sorted(np.argsort(spread)[-2:])
        ax.plot(1e3 * path[:, i], 1e3 * path[:, j], color=INK, lw=0.9, ls=(0, (4, 3)), label="shape asked for")
        ax.plot(1e3 * s["tcp_pic"][s["on_pic"], i], 1e3 * s["tcp_pic"][s["on_pic"], j], color=MUTED,
                lw=1.2, label="picture (simulated)")
        ax.plot(1e3 * s["tcp_meas"][s["on_meas"], i], 1e3 * s["tcp_meas"][s["on_meas"], j], color=BLUE,
                lw=1.4, label="measured")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"{'xyz'[i]} (mm)")
        ax.set_ylabel(f"{'xyz'[j]} (mm)")
        ax.set_title("path of the hand")
        ax.legend(fontsize=7)
    fig.suptitle(f"{name}: score {m['shown']}/10 (mean of two runs); run 1 shown, hand error "
                 f"{m['tcp_rms_mm']:.1f} mm RMS", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    savefig(fig, f"animations/{name}.png")


def report_animations(data, kin, out):
    by_name = {}
    for run in data["runs"]:
        m, s = analyse_animation(run, kin)
        m["path_xyz"] = run.get("path")
        by_name.setdefault(run["name"], []).append((m, s))
    rows = []
    for name, runs in by_name.items():
        if len(runs) > 1:
            (_, a), (_, b) = runs[0], runs[1]
            grid = np.arange(0, min(a["end"], b["end"]), 0.02)
            ta = np.column_stack([np.interp(grid, a["t"], a["tcp_meas"][:, k]) for k in range(3)])
            tb = np.column_stack([np.interp(grid, b["t"], b["tcp_meas"][:, k]) for k in range(3)])
            repeat = 1e3 * rms(np.linalg.norm(ta - tb, axis=1))
        else:
            repeat = float("nan")
        for m, _ in runs:
            m["repeat_mm"] = repeat
            m["score"] = score(m)
        for m, _ in runs:
            m["shown"] = round(float(np.mean([x["score"] for x, _ in runs])), 1)
        plot_animation(name, runs, kin)
        keys = [k for k, v in runs[0][0].items() if isinstance(v, (int, float)) and k not in ("repeat", "shown")]
        row = {"name": name, "kind": runs[0][0]["kind"],
               **{k: float((np.max if k.endswith("max_mm") else np.mean)([m[k] for m, _ in runs]))
                  for k in keys},
               "joint_rms_mrad": np.mean([m["joint_rms_mrad"] for m, _ in runs], axis=0).tolist(),
               "joint_max_mrad": np.max([m["joint_max_mrad"] for m, _ in runs], axis=0).tolist()}
        row["score"] = round(row["score"], 1)
        rows.append(row)
    out["animations"] = rows
    out["animation_summary"] = {
        "n": len(rows), "score_mean": float(np.mean([r["score"] for r in rows])),
        "score_min": float(min(r["score"] for r in rows)), "score_max": float(max(r["score"] for r in rows)),
        "tcp_rms_mean_mm": float(np.mean([r["tcp_rms_mm"] for r in rows])),
        "tcp_max_mm": float(max(r["tcp_max_mm"] for r in rows)),
        "tcp_rms_at_lag_mean_mm": float(np.mean([r["tcp_rms_at_lag_mm"] for r in rows])),
        "arm_rms_mean_mrad": float(np.mean([r["arm_rms_mrad"] for r in rows])),
        "lag_mean_s": float(np.mean([r["lag_s"] for r in rows])),
        "lag_min_s": float(min(r["lag_s"] for r in rows)), "lag_max_s": float(max(r["lag_s"] for r in rows)),
        "reach_min": float(min(r["reach"] for r in rows)),
        "repeat_mean_mm": float(np.mean([r["repeat_mm"] for r in rows])),
        "joint_rms_mean_mrad": np.mean([r["joint_rms_mrad"] for r in rows], axis=0).tolist(),
        "commands_hz": float(np.mean([r["commands_hz"] for r in rows])),
        "relative_mean": float(np.mean([r["relative"] for r in rows])),
        "by_kind": {kind: {"score": float(np.mean([r["score"] for r in rows if r["kind"] == kind])),
                           "tcp_rms_mm": float(np.mean([r["tcp_rms_mm"] for r in rows if r["kind"] == kind])),
                           "tcp_max_mm": float(max(r["tcp_max_mm"] for r in rows if r["kind"] == kind)),
                           "relative": float(np.mean([r["relative"] for r in rows if r["kind"] == kind])),
                           "arm_rms_mrad": float(np.mean([r["arm_rms_mrad"] for r in rows if r["kind"] == kind])),
                           "stretch": float(np.mean([r["played_s"] / r["designed_s"] for r in rows
                                                     if r["kind"] == kind]))}
                    for kind in ("shape", "gesture")},
    }
    shapes = [r for r in rows if r["kind"] == "shape"]
    if shapes:
        out["animation_summary"]["path_error_mean_mm"] = float(np.mean([r["path_error_mm"] for r in shapes]))
        out["animation_summary"]["path_error_picture_mean_mm"] = float(
            np.mean([r["path_error_picture_mm"] for r in shapes]))

    ordered = sorted(rows, key=lambda r: r["score"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    names = [r["name"] for r in ordered]
    for ax, key, title, fmt in zip(
            axes, ("score", "tcp_rms_mm", "tcp_rms_at_lag_mm"),
            ("Score (0 to 10)", "Hand error against the picture (mm RMS)",
             "Hand error with the delay removed (mm RMS)"), ("{:.1f}", "{:.1f}", "{:.1f}")):
        values = [r[key] for r in ordered]
        ax.barh(names, values, height=0.62, color=[BLUE if r["kind"] == "shape" else ORANGE for r in ordered])
        for y, value in enumerate(values):
            ax.text(value, y, " " + fmt.format(value), va="center", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, 10.8)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
                   labels=["traced shape", "library gesture"], fontsize=7, loc="lower right")
    savefig(fig, "animation_scores.png")

    # how the hand error depends on how fast the picture is moving, over every sample of every run
    speed, error = [], []
    for runs in by_name.values():
        for _, s in runs:
            v = np.linalg.norm(np.gradient(s["tcp_pic"], s["t"], axis=0), axis=1)
            speed.extend(v[s["keep"]])
            error.extend(s["tcp_err"][s["keep"]])
    speed, error = np.array(speed), np.array(error)
    slope = float(np.sum(speed * error) / np.sum(speed * speed))
    out["animation_summary"]["error_per_speed_ms"] = slope  # mm per (m/s) = ms
    out["animation_summary"]["error_at_rest_mm"] = float(np.median(error[speed < 0.01]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(speed, error, s=3, color=BLUE, alpha=0.25, lw=0)
    xs = np.array([0, speed.max()])
    ax.plot(xs, slope * xs, color=INK, lw=1, label=f"error = {slope:.0f} ms × hand speed")
    ax.set_xlabel("speed of the hand in the picture (m/s)")
    ax.set_ylabel("hand error (mm)")
    ax.set_title("The faster the picture moves, the further behind the arm is")
    ax.legend(fontsize=8)
    savefig(fig, "error_vs_speed.png")

    for kind, cols, file in (("shape", 3, "shape_gallery.png"),):
        items = [(n, r) for n, r in by_name.items() if r[0][0]["kind"] == kind]
        fig, axes = plt.subplots(math.ceil(len(items) / cols), cols, figsize=(11, 3.6 * math.ceil(len(items) / cols)))
        for ax, (name, runs) in zip(axes.flat, items):
            m, s = runs[0]
            path = np.array(m["path_xyz"])
            i, j = sorted(np.argsort(np.ptp(path, axis=0))[-2:])
            ax.plot(1e3 * path[:, i], 1e3 * path[:, j], color=INK, lw=0.9, ls=(0, (4, 3)), label="shape asked for")
            ax.plot(1e3 * s["tcp_pic"][s["on_pic"], i], 1e3 * s["tcp_pic"][s["on_pic"], j], color=MUTED,
                    lw=1.2, label="picture (simulated)")
            ax.plot(1e3 * s["tcp_meas"][s["on_meas"], i], 1e3 * s["tcp_meas"][s["on_meas"], j], color=BLUE,
                    lw=1.4, label="measured")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{name}  ({m['path_error_mm']:.1f} mm off the shape)", fontsize=9)
            ax.set_xlabel(f"{'xyz'[i]} (mm)")
            ax.set_ylabel(f"{'xyz'[j]} (mm)")
        axes.flat[0].legend(fontsize=7)
        for ax in axes.flat[len(items):]:
            ax.set_visible(False)
        fig.tight_layout()
        savefig(fig, file)

    items = [(n, r) for n, r in by_name.items() if r[0][0]["kind"] == "gesture"]
    fig, axes = plt.subplots(math.ceil(len(items) / 2), 2, figsize=(12, 2.3 * math.ceil(len(items) / 2)))
    for ax, (name, runs) in zip(axes.flat, items):
        m, s = runs[0]
        for k, joint in enumerate(JOINTS):
            ax.plot(s["pt"], s["pq"][:, k], color=JOINT_COLOR[joint], lw=0.9, ls=(0, (3, 2)))
            ax.plot(s["t"], s["q"][:, k], color=JOINT_COLOR[joint], lw=1.4, label=joint)
        ax.set_xlim(0, s["end"] + 0.6)
        ax.set_title(f"{name}  (score {m['shown']})", fontsize=9)
        ax.set_ylabel("rad")
    axes.flat[0].legend(fontsize=7, ncol=4, title="solid = measured, dashed = picture", title_fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    savefig(fig, "gesture_gallery.png")
    for runs in by_name.values():
        for m, _ in runs:
            m.pop("path_xyz", None)


# --- tables -----------------------------------------------------------------------

def tables(out):
    lines = []

    def table(title, header, rows):
        lines.extend(["", f"### {title}", "", "| " + " | ".join(header) + " |",
                      "|" + "|".join(" --- " for _ in header) + "|"])
        lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)

    if "static" in out:
        table("Static accuracy, one joint at a time (mrad)",
              ["joint", "mean error", "RMS", "worst", "up − down", "repeatability (1σ)", "noise at rest", "n"],
              [[n, f"{r['bias_mrad']:+.1f}", f"{r['rms_mrad']:.1f}", f"{r['max_mrad']:.1f}",
                f"{r['hysteresis_mrad']:+.1f}", f"{r['repeat_mrad']:.2f}", f"{r['noise_mrad']:.2f}", r["n"]]
               for n, r in out["static"].items()])
    if "step" in out:
        table("0.5 rad step, mean of both directions",
              ["joint and servo setting", "dead time (ms)", "10–90% rise (s)", "peak speed (rad/s)",
               "overshoot (mrad)", "settled within 10 mrad (s)", "final error (mrad)"],
              [[k, f"{1e3 * r['dead_s']:.0f}", f"{r['rise_s']:.2f}", f"{r['peak_speed']:.2f}",
                f"{r['overshoot_mrad']:.1f}", f"{r['settle_s']:.2f}", f"{r['final_error_mrad']:+.1f}"]
               for k, r in out["step"]["by_profile"].items()])
    if "sine" in out:
        table("Sine tracking",
              ["joint", "± rad", "Hz", "asked peak speed (rad/s)", "gain", "lag (ms)", "RMS error (mrad)",
               "distortion"],
              [[f["joint"], f["amplitude"], f["hz"], f"{f['asked_speed']:.2f}", f"{f['gain']:.2f}",
                f"{1e3 * f['lag_s']:.0f}", f"{f['rms_error_mrad']:.0f}", f"{100 * f['distortion']:.0f}%"]
               for f in out["sine"]["fits"]])
    if "stream" in out:
        table("One reference, different ways of sending it",
              ["variant", "commands per second", "RMS error (mrad)", "delay (ms)",
               "RMS error at that delay (mrad)", "speed ripple (rad/s)"],
              [[r["label"], f"{r['commands_hz']:.0f}", f"{r['rms_mrad']:.0f}", f"{1e3 * r['lag_s']:.0f}",
                f"{r['rms_at_lag_mrad']:.1f}", f"{r['speed_ripple']:.2f}"] for r in out["stream"]])
    if "limits" in out:
        table("A sine on the base at ±0.15 rad, by servo setting",
              ["Hz", "speed", "acc", "gain", "lag (ms)", "distortion", "RMS error (mrad)"],
              [[r["hz"], r["speed"], r["acc"], f"{r['gain']:.2f}", f"{1e3 * r['lag_s']:.0f}",
                f"{100 * r['distortion']:.0f}%", f"{r['rms_error_mrad']:.0f}"] for r in out["limits"]])
    if "hold" in out:
        table("Standing still: movement of each joint (mrad peak to peak)",
              ["shoulder (rad)", "elbow (rad)", "base", "shoulder", "elbow", "gripper", "shoulder frequency (Hz)",
               "shoulder load"],
              [[f"{r['shoulder']:.2f}", f"{r['elbow']:.2f}", f"{r['base_pp_mrad']:.1f}",
                f"{r['shoulder_pp_mrad']:.1f}", f"{r['elbow_pp_mrad']:.1f}", f"{r['gripper_pp_mrad']:.1f}",
                f"{r['shoulder_hz']:.1f}" if r["shoulder_pp_mrad"] > 4e3 * COUNT else "",
                f"{r['load'][1]:.0f}"] for r in out["hold"]["rows"]])
    if "animations" in out:
        table("Animations (mean of two runs; worst is the worst of both)",
              ["animation", "kind", "designed (s)", "played (s)", "arm joints RMS (mrad)", "hand RMS (mm)",
               "of the motion's size", "hand worst (mm)", "delay (ms)", "hand RMS, delay removed (mm)", "reach kept",
               "run-to-run (mm)", "off the shape (mm)", "score"],
              [[r["name"], r["kind"], f"{r['designed_s']:.1f}", f"{r['played_s']:.1f}",
                f"{r['arm_rms_mrad']:.0f}", f"{r['tcp_rms_mm']:.1f}", f"{100 * r['relative']:.1f}%",
                f"{r['tcp_max_mm']:.1f}",
                f"{1e3 * r['lag_s']:.0f}", f"{r['tcp_rms_at_lag_mm']:.1f}", f"{100 * r['reach']:.0f}%",
                f"{r['repeat_mm']:.1f}", f"{r['path_error_mm']:.1f}" if "path_error_mm" in r else "",
                f"{r['score']:.1f}"]
               for r in sorted(out["animations"], key=lambda r: -r["score"])])
    with open(os.path.join(PAPER_DIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("# Result tables\n\nGenerated by `python -m roarm_rl.benchmark_report`; do not edit by hand.\n")
        f.write("\n".join(lines) + "\n")


def main():
    kin = Kin()
    out = {}
    logs = {name: load(name) for name in ("static", "poses", "step", "sine", "stream", "limits", "hold",
                                          "iksim", "ik", "animations")}
    if logs["static"]:
        report_static(logs["static"], out)
    if logs["poses"]:
        report_poses(logs["poses"], kin, out)
    if logs["step"]:
        report_step(logs["step"], out)
    if logs["sine"]:
        report_sine(logs["sine"], out)
    if logs["stream"]:
        report_stream(logs["stream"], out)
    if logs["limits"]:
        report_limits(logs["limits"], out)
    if logs["hold"]:
        report_hold(logs["hold"], out)
    if logs["iksim"]:
        report_iksim(logs["iksim"], out)
    if logs["ik"]:
        report_ik(logs["ik"], kin, out)
    settled_logs = [logs[k] for k in ("static", "poses", "ik") if logs[k]]
    if settled_logs:
        report_firmware_model(settled_logs, kin, out)
    if logs["animations"]:
        report_animations(logs["animations"], kin, out)
    with open(os.path.join(PAPER_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    tables(out)
    print("figures in", FIG_DIR)


if __name__ == "__main__":
    main()
