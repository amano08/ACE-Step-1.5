"""Secondary generation-tab controls (cover, custom prompt, repaint)."""

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t

from .user_defaults import (
    audio_cover_strength_default,
    caption_default,
    cover_noise_strength_default,
    reference_audio_default,
    remix_only,
)

_DEFAULT_CAPTION_FILE = Path(__file__).resolve().parents[4] / "default_caption.txt"


def _default_caption() -> str:
    """Prefill the caption box from ACESTEP_DEFAULT_CAPTION_FILE, else default_caption.txt."""
    return caption_default(_DEFAULT_CAPTION_FILE)


def build_cover_strength_controls() -> dict[str, Any]:
    """Create code/remix strength controls used by non-simple generation modes.

    Args:
        None.

    Returns:
        A component map containing audio/code strength sliders and remix help group.
    """

    # Mirrors the Remix branch of compute_mode_ui_updates at build time, since the
    # mode radio that would normally push those updates is hidden.
    only_remix = remix_only()
    audio_cover_strength = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=audio_cover_strength_default(0.35),
        step=0.01,
        label=t("generation.remix_strength_label") if only_remix else t("generation.codes_strength_label"),
        info=t("generation.remix_strength_info") if only_remix else t("generation.codes_strength_info"),
        elem_classes=["has-info-container"],
        visible=True,
    )
    with gr.Group(visible=only_remix) as remix_help_group:
        create_help_button("generation_remix")
        no_fsq = gr.Checkbox(
            label=t("generation.no_fsq_label"),
            value=False,
            info=t("generation.no_fsq_info"),
            elem_classes=["has-info-container"],
        )
    cover_noise_strength = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=cover_noise_strength_default(0.5),
        step=0.01,
        label=t("generation.cover_noise_strength_label"),
        info=t("generation.cover_noise_strength_info"),
        elem_classes=["has-info-container"],
        visible=only_remix,
    )
    return {
        "audio_cover_strength": audio_cover_strength,
        "remix_help_group": remix_help_group,
        "no_fsq": no_fsq,
        "cover_noise_strength": cover_noise_strength,
    }


def build_custom_mode_controls() -> dict[str, Any]:
    """Create custom-mode caption, lyrics, and reference-audio controls.

    Args:
        None.

    Returns:
        A component map containing custom-mode text/audio inputs and formatting actions.
    """

    with gr.Group(visible=True, elem_classes=["has-info-container"]) as custom_mode_group:
        create_help_button("generation_custom")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=200):
                reference_audio = gr.Audio(
                    label=t("generation.reference_audio"),
                    type="filepath",
                    show_label=True,
                    value=reference_audio_default(),
                )
                # gr.Audio has no ``info=`` parameter (gradio 6.2), so the
                # explanation goes in a caption below the widget.
                gr.Markdown(
                    t("generation.reference_audio_info"),
                    elem_classes=["audio-hint"],
                )
            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        captions = gr.Textbox(
                            label=t("generation.caption_label"),
                            placeholder=t("generation.caption_placeholder"),
                            value=_default_caption(),
                            info=t("generation.caption_info"),
                            elem_classes=["has-info-container"],
                            lines=12,
                            max_lines=12,
                        )
                        with gr.Row(elem_classes="instrumental-row"):
                            format_caption_btn = gr.Button(
                                t("generation.format_caption_btn"),
                                variant="secondary",
                                size="sm",
                            )
                    with gr.Column(scale=1):
                        lyrics = gr.Textbox(
                            label=t("generation.lyrics_label"),
                            placeholder=t("generation.lyrics_placeholder"),
                            value="[Instrumental]",
                            info=t("generation.lyrics_info"),
                            elem_classes=["has-info-container"],
                            lines=12,
                            max_lines=12,
                        )
                        with gr.Row(elem_classes="instrumental-row"):
                            instrumental_checkbox = gr.Checkbox(
                                label=t("generation.instrumental_label"),
                                value=True,
                                info=t("generation.instrumental_info"),
                                elem_classes=["has-info-container"],
                                scale=1,
                            )
                            format_lyrics_btn = gr.Button(
                                t("generation.format_lyrics_btn"),
                                variant="secondary",
                                size="sm",
                                scale=2,
                            )
            with gr.Column(scale=1, min_width=80, elem_classes="icon-btn-wrap"):
                sample_btn = gr.Button(t("generation.sample_btn"), variant="primary", size="lg")
    return {
        "custom_mode_group": custom_mode_group,
        "reference_audio": reference_audio,
        "captions": captions,
        "format_caption_btn": format_caption_btn,
        "lyrics": lyrics,
        "instrumental_checkbox": instrumental_checkbox,
        "format_lyrics_btn": format_lyrics_btn,
        "sample_btn": sample_btn,
    }


def build_repainting_controls() -> dict[str, Any]:
    """Create repainting range controls used by repaint/lego flows.

    Args:
        None.

    Returns:
        A component map containing repainting group, header, and start/end controls.
    """

    with gr.Group(visible=False) as repainting_group:
        create_help_button("generation_repaint")
        repainting_header_html = gr.HTML(f"<h5>{t('generation.repainting_controls')}</h5>")
        with gr.Row():
            repainting_start = gr.Number(
                label=t("generation.repainting_start"),
                value=0.0,
                step=0.1,
            )
            repainting_end = gr.Number(
                label=t("generation.repainting_end"),
                value=-1,
                minimum=-1,
                step=0.1,
            )
        with gr.Row():
            repaint_mode = gr.Dropdown(
                label="Repaint Mode",
                choices=["conservative", "balanced", "aggressive"],
                value="balanced",
                info="conservative=preserve source, aggressive=full regeneration",
            )
            repaint_strength = gr.Slider(
                label="Repaint Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.5,
                info="0=conservative, 1=aggressive (balanced mode only)",
            )
        repaint_strength_memory = gr.State(value=0.5)
    return {
        "repainting_group": repainting_group,
        "repainting_header_html": repainting_header_html,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "repaint_mode": repaint_mode,
        "repaint_strength": repaint_strength,
        "repaint_strength_memory": repaint_strength_memory,
    }


