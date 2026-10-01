#!/usr/bin/env python3
"""Prove the Stop flag actually aborts an in-flight diffusion run.

Two checks, because the two loops the flag guards are separate code:

A. sft main diffusion loop (CEO's configured renderer): start a long run, raise
   the flag mid-flight, and require the run to abort rather than finish.
B. flow-edit loop: raise the flag *before* starting, and require the run to
   abort instead of producing audio -- proves the check is wired into that loop.

Then a clean run to confirm ``clear_stop`` un-sticks the flag.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from acestep.generation_stop import GenerationStopped, clear_stop, request_stop
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "stop_verify")
SRC = os.path.join(PROJECT_ROOT, "output", "flowedit_test",
                   "bf121599-7a43-22d0-7c48-d274c456d549.wav")

SRC_CAPTION = "Japanese anime opening theme, energetic J-rock band, distorted electric guitars."
TGT_CAPTION = "Instrumental big band jazz, trumpets trombones saxophones, swinging, no vocals."

STOP_AFTER_S = 20.0
results = {}


def _load(config_path):
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root=PROJECT_ROOT, config_path=config_path,
        device="auto", offload_to_cpu=False,
    )
    if not ok:
        print(f"DiT init failed ({config_path}): {msg}")
        sys.exit(1)
    return dit


def _run(dit, params, cfg):
    """Return (aborted, elapsed_seconds)."""
    t0 = time.time()
    try:
        res = generate_music(dit, None, params=params, config=cfg, save_dir=SAVE_DIR)
        return (not res.success), (time.time() - t0), res
    except GenerationStopped:
        return True, (time.time() - t0), None


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = GenerationConfig(batch_size=1, audio_format="wav",
                           use_random_seed=False, seeds=[12345])

    long_params = lambda: GenerationParams(
        task_type="text2music", thinking=False,
        caption=TGT_CAPTION, lyrics="[inst]",
        bpm=170, keyscale="D minor", timesignature="4",
        duration=120, inference_steps=60, guidance_scale=3.0, seed=12345,
    )

    # ---- A: sft, stopped mid-flight ----------------------------------
    sft = _load("acestep-v15-sft")
    clear_stop()
    _, baseline, base_res = _run(sft, long_params(), cfg)
    print(f"[A] baseline uninterrupted: {baseline:.1f}s (success={base_res.success})")
    if baseline < STOP_AFTER_S * 1.5:
        print(f"[A] INCONCLUSIVE: baseline {baseline:.1f}s too short vs stop at {STOP_AFTER_S}s")
        results["A"] = False
    else:
        clear_stop()
        threading.Timer(STOP_AFTER_S, request_stop).start()
        aborted, elapsed, _ = _run(sft, long_params(), cfg)
        print(f"[A] stopped run: aborted={aborted}, {elapsed:.1f}s vs baseline {baseline:.1f}s")
        results["A"] = aborted and elapsed < baseline * 0.9

    # ---- A2: flag cleared -> full run succeeds again ------------------
    clear_stop()
    _, after_s, after_res = _run(sft, long_params(), cfg)
    results["A2"] = bool(after_res and after_res.success)
    print(f"[A2] after clear_stop: success={results['A2']} ({after_s:.1f}s)")
    del sft

    # ---- B: flow-edit loop, flag raised before start ------------------
    turbo = _load("acestep-v15-turbo")
    morph = GenerationParams(
        task_type="text2music", thinking=False,
        caption=TGT_CAPTION, lyrics="[inst]", src_audio=SRC,
        flow_edit_morph=True,
        flow_edit_source_caption=SRC_CAPTION, flow_edit_source_lyrics="[inst]",
        flow_edit_n_min=0.0, flow_edit_n_max=1.0, flow_edit_n_avg=1,
        shift=3.0, bpm=170, keyscale="D minor", timesignature="4",
        duration=30, inference_steps=8, guidance_scale=1.0, seed=12345,
    )
    request_stop()
    aborted_b, elapsed_b, _ = _run(turbo, morph, cfg)
    results["B"] = aborted_b
    print(f"[B] flow-edit with flag pre-raised: aborted={aborted_b} ({elapsed_b:.1f}s)")

    clear_stop()
    _, _, b_after = _run(turbo, morph, cfg)
    results["B2"] = bool(b_after and b_after.success)
    print(f"[B2] flow-edit after clear_stop: success={results['B2']}")

    print("=" * 60)
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
