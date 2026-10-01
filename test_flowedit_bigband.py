#!/usr/bin/env python3
"""Verify the Flow-Edit morph path can transplant orchestration (J-rock -> big band).

Step 1: synthesize a 30s J-rock anime-OP instrumental as the "original song".
Step 2: morph it toward a big band caption via flow_edit_morph on text2music.
Step 3: also run a plain Remix (cover) for A/B comparison at CEO's failing settings.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "flowedit_test")

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


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger.info("Loading turbo DiT (no LM needed: flow-edit skips it)...")
    t0 = time.time()
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=False,
    )
    if not ok:
        logger.error(f"DiT init failed: {msg}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s")

    cfg = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[12345])

    # --- Step 1: make the "original song" -------------------------------
    logger.info("=== step 1: synthesize J-rock source ===")
    src_params = GenerationParams(
        task_type="text2music",
        thinking=False,
        caption=SRC_CAPTION,
        lyrics="[inst]",
        bpm=170,
        keyscale="D minor",
        timesignature="4",
        duration=30,
        inference_steps=8,
        guidance_scale=1.0,
        seed=12345,
    )
    t0 = time.time()
    res = generate_music(dit, None, params=src_params, config=cfg, save_dir=SAVE_DIR)
    if not res.success:
        logger.error(f"source generation FAILED: {res.status_message}")
        sys.exit(1)
    src_path = res.audios[0]["path"]
    logger.info(f"source OK ({time.time() - t0:.1f}s) -> {src_path}")

    # --- Step 2: flow-edit morph toward big band -------------------------
    for n_max in (1.0, 0.8):
        logger.info(f"=== step 2: flow-edit morph (n_max={n_max}) ===")
        morph_params = GenerationParams(
            task_type="text2music",
            thinking=False,
            caption=TGT_CAPTION,
            lyrics="[inst]",
            src_audio=src_path,
            flow_edit_morph=True,
            flow_edit_source_caption=SRC_CAPTION,
            flow_edit_source_lyrics="[inst]",
            flow_edit_n_min=0.0,
            flow_edit_n_max=n_max,
            flow_edit_n_avg=1,
            shift=3.0,
            bpm=170,
            keyscale="D minor",
            timesignature="4",
            duration=30,
            inference_steps=8,
            guidance_scale=1.0,
            seed=12345,
        )
        t0 = time.time()
        res = generate_music(dit, None, params=morph_params, config=cfg, save_dir=SAVE_DIR)
        if res.success:
            logger.info(f"morph n_max={n_max} OK ({time.time() - t0:.1f}s) -> {res.audios[0]['path']}")
        else:
            logger.error(f"morph n_max={n_max} FAILED: {res.status_message}")

    # --- Step 3: plain Remix at CEO's failing settings, for A/B ----------
    for acs, cns in ((0.5, 0.5), (0.25, 0.0)):
        logger.info(f"=== step 3: plain cover (cover_strength={acs}, noise={cns}) ===")
        cover_params = GenerationParams(
            task_type="cover",
            thinking=False,
            caption=TGT_CAPTION,
            lyrics="[inst]",
            src_audio=src_path,
            audio_cover_strength=acs,
            cover_noise_strength=cns,
            bpm=170,
            keyscale="D minor",
            timesignature="4",
            duration=30,
            inference_steps=8,
            guidance_scale=1.0,
            seed=12345,
        )
        t0 = time.time()
        res = generate_music(dit, None, params=cover_params, config=cfg, save_dir=SAVE_DIR)
        if res.success:
            logger.info(f"cover {acs}/{cns} OK ({time.time() - t0:.1f}s) -> {res.audios[0]['path']}")
        else:
            logger.error(f"cover {acs}/{cns} FAILED: {res.status_message}")

    logger.info("Done. Output dir: " + SAVE_DIR)


if __name__ == "__main__":
    main()
