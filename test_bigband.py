#!/usr/bin/env python3
"""CEO smoke test: anime-style tune arranged for big band / wind ensemble (instrumental)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "bigband_test")

CASES = [
    {
        "name": "bigband",
        "caption": (
            "An instrumental big band jazz arrangement of an upbeat Japanese anime "
            "opening theme. Full horn section with trumpets, trombones and saxophones "
            "playing tight shout-chorus harmonies, walking upright bass, swinging ride "
            "cymbal and brushed snare, comping piano. Bright, triumphant, virtuosic, "
            "with a tenor sax solo over the bridge. Studio big band recording, no vocals."
        ),
        "bpm": 168,
        "keyscale": "F major",
    },
    {
        "name": "windensemble",
        "caption": (
            "An instrumental concert wind ensemble (symphonic band) arrangement of a "
            "heroic Japanese video game theme. Flutes and clarinets carry the melody, "
            "answered by french horns and euphonium, supported by tuba and timpani, "
            "with cymbal crashes at the climax. Lyrical, cinematic, warm acoustic "
            "concert hall recording, no vocals."
        ),
        "bpm": 96,
        "keyscale": "Eb major",
    },
]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger.info("Initializing DiT handler (turbo)...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=False,
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s")

    logger.info("Initializing LLM handler (0.6B, MLX)...")
    t0 = time.time()
    llm_handler = LLMHandler()
    status_msg, success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="mlx",
        device="auto",
        offload_to_cpu=False,
        dtype=None,
    )
    if not success:
        logger.error(f"LLM init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"LLM loaded in {time.time() - t0:.1f}s")

    for case in CASES:
        logger.info(f"=== generating {case['name']} ===")
        params = GenerationParams(
            task_type="text2music",
            thinking=True,
            caption=case["caption"],
            lyrics="[inst]",
            bpm=case["bpm"],
            keyscale=case["keyscale"],
            timesignature="4",
            vocal_language="en",
            duration=60,
            inference_steps=8,
            guidance_scale=1.0,
            seed=-1,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")

        t0 = time.time()
        result = generate_music(
            dit_handler, llm_handler, params=params, config=config, save_dir=SAVE_DIR
        )
        elapsed = time.time() - t0

        if result.success:
            logger.info(f"{case['name']} OK — {elapsed:.1f}s for 60s audio")
            for audio in result.audios:
                logger.info(f"  -> {audio.get('path', '(in-memory)')}")
        else:
            logger.error(f"{case['name']} FAILED — {result.status_message}")

    logger.info("Done. Output dir: " + SAVE_DIR)


if __name__ == "__main__":
    main()
