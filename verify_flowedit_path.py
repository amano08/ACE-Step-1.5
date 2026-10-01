#!/usr/bin/env python3
"""Prove the flow-edit V_delta path actually engages in a real run.

Re-runs one morph over the already-generated J-rock source and asserts the
engine logged the V_delta branch, so the earlier A/B files are trustworthy.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "flowedit_verify")
SRC = os.path.join(PROJECT_ROOT, "output", "flowedit_test",
                   "bf121599-7a43-22d0-7c48-d274c456d549.wav")

SRC_CAPTION = (
    "Japanese anime opening theme, energetic J-rock band, distorted electric guitars, "
    "driving rock drum kit, electric bass, bright synth pads, fast and heroic."
)
TGT_CAPTION = (
    "Instrumental big band jazz. Four trumpets, four trombones, five saxophones playing "
    "tight shout-chorus harmonies and punchy section hits. Comping piano, walking upright "
    "bass, swinging ride cymbal and snare. Bright, triumphant, studio big band recording, "
    "no vocals, no electric guitar, no synth."
)

SENTINEL = "encoding src_audio for V_delta integration"

seen = {"hit": False}
logger.add(lambda m: seen.__setitem__("hit", seen["hit"] or SENTINEL in m), level="INFO")


def main():
    assert os.path.exists(SRC), f"source audio missing: {SRC}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root=PROJECT_ROOT, config_path="acestep-v15-turbo",
        device="auto", offload_to_cpu=False,
    )
    if not ok:
        logger.error(f"DiT init failed: {msg}")
        sys.exit(1)

    cfg = GenerationConfig(batch_size=1, audio_format="wav",
                           use_random_seed=False, seeds=[12345])
    params = GenerationParams(
        task_type="text2music", thinking=False,
        caption=TGT_CAPTION, lyrics="[inst]", src_audio=SRC,
        flow_edit_morph=True,
        flow_edit_source_caption=SRC_CAPTION, flow_edit_source_lyrics="[inst]",
        flow_edit_n_min=0.0, flow_edit_n_max=1.0, flow_edit_n_avg=1,
        shift=3.0, bpm=170, keyscale="D minor", timesignature="4",
        duration=30, inference_steps=8, guidance_scale=1.0, seed=12345,
    )
    t0 = time.time()
    res = generate_music(dit, None, params=params, config=cfg, save_dir=SAVE_DIR)

    print("=" * 60)
    print(f"success        : {res.success}")
    print(f"elapsed        : {time.time() - t0:.1f}s")
    print(f"V_delta branch : {'HIT' if seen['hit'] else 'NOT REACHED'}")
    if res.success:
        print(f"output         : {res.audios[0]['path']}")
    else:
        print(f"status         : {res.status_message}")
    print("=" * 60)
    sys.exit(0 if (res.success and seen["hit"]) else 1)


if __name__ == "__main__":
    main()
