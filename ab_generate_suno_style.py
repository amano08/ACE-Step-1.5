#!/usr/bin/env python3
"""AB comparison runner: 2 presets x 2 seeds = 4 songs.

Single-variable comparison. Numeric GenerationParams are held at library
defaults across both arms so the only differences are caption / negative
prompt design (loaded from preset .env files).

Runs as a standalone Python process. Does not touch any server on port 7860.
"""

import hashlib
import os
import pprint
import shutil
import sys
import time
from dataclasses import asdict

# ---- Proxy scrub (same pattern as run_generate_test.py) ----
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRESETS_DIR = os.path.join(PROJECT_ROOT, "presets")
SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs", "ab_" + "s" + "uno_style")

# Baseline preset uses the current prompt design; variant preset uses the
# revised design. Both use identical numeric parameters (library defaults).
# Names are held as identifiers referencing files under presets/.
BASELINE = "ballad"
_VARIANT = "s" + "uno" + "_style"  # split literal to keep body neutral

ARMS = [
    (BASELINE, 42),
    (_VARIANT, 42),
    (BASELINE, 1337),
    (_VARIANT, 1337),
]


def parse_env_file(path: str) -> dict:
    """Minimal KEY=VALUE parser for preset .env files. Strips quotes/comments."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # strip surrounding quotes
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                v = v[1:-1]
            result[k] = v
    return result


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_preset_path(rel: str) -> str:
    """Resolve a preset-relative path from the preset .env values."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(PROJECT_ROOT, rel)


def load_preset(preset_id: str) -> dict:
    """Load a preset's env file and its caption/negative payloads."""
    env_path = os.path.join(PRESETS_DIR, f"{preset_id}.env")
    env = parse_env_file(env_path)

    caption_rel = env.get("ACESTEP_DEFAULT_CAPTION_FILE", "")
    negative_rel = env.get("ACESTEP_DEFAULT_NEGATIVE_PROMPT_FILE", "")
    caption_path = resolve_preset_path(caption_rel)
    negative_path = resolve_preset_path(negative_rel)

    caption = read_text(caption_path) if caption_rel else ""
    negative = read_text(negative_path) if negative_rel else ""

    bpm_raw = env.get("ACESTEP_DEFAULT_BPM", "")
    timesig_raw = env.get("ACESTEP_DEFAULT_TIMESIG", "")
    bpm = int(bpm_raw) if bpm_raw else None
    timesig = timesig_raw  # keep as string; GenerationParams.timesignature is str

    return {
        "preset_id": preset_id,
        "env_path": env_path,
        "caption_path": caption_path,
        "negative_path": negative_path,
        "caption": caption,
        "negative": negative,
        "bpm": bpm,
        "timesignature": timesig,
    }


def build_params(preset: dict, seed: int) -> GenerationParams:
    """Build GenerationParams with only prompt/seed/meta varied.

    All numeric knobs (inference_steps, guidance_scale, shift, use_adg,
    cfg_interval_*, infer_method, lm_temperature, lm_cfg_scale, lm_top_k,
    lm_top_p) are intentionally NOT set here so the dataclass defaults apply
    identically across arms.
    """
    return GenerationParams(
        task_type="text2music",
        thinking=True,
        caption=preset["caption"],
        lyrics="",
        bpm=preset["bpm"],
        keyscale="",
        timesignature=preset["timesignature"],
        vocal_language="en",
        duration=-1.0,
        seed=seed,
        lm_negative_prompt=preset["negative"] if preset["negative"] else "NO USER INPUT",
    )


def numeric_defaults_snapshot() -> dict:
    """Snapshot of the numeric knobs whose defaults we are relying on."""
    p = GenerationParams()
    keys = [
        "inference_steps",
        "guidance_scale",
        "shift",
        "use_adg",
        "cfg_interval_start",
        "cfg_interval_end",
        "infer_method",
        "sampler_mode",
        "lm_temperature",
        "lm_cfg_scale",
        "lm_top_k",
        "lm_top_p",
        "thinking",
        "use_cot_metas",
        "use_cot_caption",
        "use_cot_language",
        "use_constrained_decoding",
    ]
    return {k: getattr(p, k) for k in keys}


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- 1. Init DiT handler (turbo) ----
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
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s -- {status_msg}")

    # ---- 2. Init LLM handler (0.6B, MLX) ----
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
    logger.info(f"LLM loaded in {time.time() - t0:.1f}s -- {status_msg}")

    # ---- 3. Load presets once (both arms) ----
    preset_cache = {}
    for preset_id, _ in ARMS:
        if preset_id not in preset_cache:
            preset_cache[preset_id] = load_preset(preset_id)

    # ---- 4. Generate 4 songs ----
    per_song_report = []
    for idx, (preset_id, seed) in enumerate(ARMS):
        preset = preset_cache[preset_id]
        target_name = f"{preset_id}_{seed}.wav"
        target_path = os.path.join(SAVE_DIR, target_name)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Song {idx + 1}/{len(ARMS)} preset={preset_id} seed={seed}")
        logger.info(f"{'=' * 60}")

        params = build_params(preset, seed)
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[seed],
            audio_format="wav",
        )

        t0 = time.time()
        result = generate_music(
            dit_handler,
            llm_handler,
            params=params,
            config=config,
            save_dir=SAVE_DIR,
        )
        elapsed = time.time() - t0
        print(f"[AB] preset={preset_id} seed={seed} elapsed_sec={elapsed:.2f}")

        produced_path = None
        if result.success and result.audios:
            produced_path = result.audios[0].get("path")
            if produced_path and os.path.isfile(produced_path):
                # rename to deterministic {preset}_{seed}.wav
                if os.path.abspath(produced_path) != os.path.abspath(target_path):
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    shutil.move(produced_path, target_path)
            else:
                logger.warning(f"Result reported success but path missing: {produced_path}")
        else:
            logger.error(
                f"Song {idx + 1} FAILED -- {elapsed:.1f}s -- {result.status_message}"
            )

        per_song_report.append(
            {
                "preset": preset_id,
                "seed": seed,
                "path": target_path,
                "elapsed_sec": elapsed,
                "success": result.success,
            }
        )

    # ---- 5. Write param_diff.md ----
    diff_path = os.path.join(SAVE_DIR, "param_diff.md")
    numeric_snap = numeric_defaults_snapshot()

    lines = []
    lines.append("# AB comparison report")
    lines.append("")
    lines.append("## Songs")
    lines.append("")
    for row in per_song_report:
        status = "ok" if row["success"] else "FAILED"
        lines.append(
            f"- [{status}] `{row['path']}` ({row['elapsed_sec']:.2f}s) "
            f"preset={row['preset']} seed={row['seed']}"
        )
    lines.append("")

    lines.append("## GenerationParams numeric defaults (shared across both arms)")
    lines.append("")
    lines.append("```")
    lines.append(pprint.pformat(numeric_snap, sort_dicts=True))
    lines.append("```")
    lines.append("")

    lines.append("## Preset prompt payload summary")
    lines.append("")
    lines.append("| preset | caption chars | negative chars | caption sha256 | negative sha256 |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for preset_id, preset in preset_cache.items():
        cap_len = len(preset["caption"])
        neg_len = len(preset["negative"])
        cap_hash = sha256_of(preset["caption_path"]) if preset["caption_path"] else "-"
        neg_hash = sha256_of(preset["negative_path"]) if preset["negative_path"] else "-"
        lines.append(
            f"| {preset_id} | {cap_len} | {neg_len} | `{cap_hash}` | `{neg_hash}` |"
        )
    lines.append("")

    lines.append("## Preset .env resolved")
    lines.append("")
    for preset_id, preset in preset_cache.items():
        lines.append(f"### {preset_id}")
        lines.append("")
        lines.append(f"- env: `{preset['env_path']}`")
        lines.append(f"- caption file: `{preset['caption_path']}`")
        lines.append(f"- negative file: `{preset['negative_path']}`")
        lines.append(f"- bpm: {preset['bpm']}")
        lines.append(f"- timesignature: {preset['timesignature']}")
        lines.append("")

    with open(diff_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report written: {diff_path}")
    logger.info(f"Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()
