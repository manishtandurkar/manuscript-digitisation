from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_output_path, preprocess
from src.utils import ensure_parent_dir

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enhanced"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("app")


def discover_raw_images() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    paths: list[Path] = []
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(set(paths))


def paths_to_gallery_data(paths: list[Path]) -> list[tuple[str, str]]:
    return [(str(p), str(p.relative_to(RAW_DIR))) for p in paths]


def make_checkbox_choices(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(RAW_DIR)) for p in paths]


def process_selected(
    selected_names: list[str],
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[tuple[str, str]]]:
    if not selected_names:
        return "No images selected. Tick one or more images from the list.", []

    results: list[tuple[str, str]] = []
    log_lines: list[str] = []
    total = len(selected_names)

    for i, rel_name in enumerate(selected_names):
        try:
            progress((i + 0.5) / total, desc=f"Processing {rel_name}")
        except Exception:
            pass
        input_path = RAW_DIR / rel_name
        output_path = build_output_path(input_path, OUTPUT_DIR)
        try:
            ensure_parent_dir(output_path)
            preprocess(str(input_path), str(output_path))
            results.append((str(input_path), f"BEFORE  {rel_name}"))
            results.append((str(output_path), f"AFTER   {output_path.name}"))
            log_lines.append(f"[OK]  {rel_name}")
        except Exception as exc:
            LOGGER.exception("Failed to process %s", rel_name)
            log_lines.append(f"[ERR] {rel_name}: {exc}")

    ok_count = sum(1 for line in log_lines if line.startswith("[OK]"))
    err_count = len(log_lines) - ok_count
    summary = f"Processed {ok_count}/{len(selected_names)} image(s)"
    if err_count:
        summary += f" — {err_count} error(s)"
    status = summary + "\n\n" + "\n".join(log_lines)
    return status, results


def refresh_gallery() -> tuple[list, object, list]:
    paths = discover_raw_images()
    choices = make_checkbox_choices(paths)
    gallery_data = paths_to_gallery_data(paths)
    if not paths:
        LOGGER.info("No images found in %s", RAW_DIR)
    return gallery_data, gr.CheckboxGroup(choices=choices, value=[]), []


def build_ui() -> gr.Blocks:
    initial_paths = discover_raw_images()
    initial_gallery = paths_to_gallery_data(initial_paths)
    initial_choices = make_checkbox_choices(initial_paths)

    empty_note = "" if initial_paths else (
        f"> No images found in `data/raw/`. "
        f"Add images to `{RAW_DIR}` and click **Refresh**."
    )

    with gr.Blocks(title="Manuscript Digitisation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Manuscript Digitisation\n*Stage 1 — Image Preprocessing*")

        if empty_note:
            gr.Markdown(empty_note)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Raw Images")
                raw_gallery = gr.Gallery(
                    value=initial_gallery,
                    label="All raw images",
                    columns=3,
                    height=380,
                    allow_preview=True,
                    object_fit="contain",
                    show_label=False,
                )

                gr.Markdown("### Select images to process")
                selected_images = gr.CheckboxGroup(
                    choices=initial_choices,
                    value=[],
                    label="Images",
                    show_label=False,
                )

                with gr.Row():
                    select_all_btn = gr.Button("Select All", size="sm")
                    deselect_btn = gr.Button("Deselect All", size="sm")

                with gr.Row():
                    process_btn = gr.Button("Process Selected", variant="primary")
                    refresh_btn = gr.Button("Refresh", variant="secondary")

                status_box = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    placeholder="Processing status will appear here…",
                )

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                gr.Markdown(
                    "*Images are shown in pairs: **BEFORE** (left) | **AFTER** (right).*"
                )
                results_gallery = gr.Gallery(
                    label="Results",
                    columns=2,
                    height=600,
                    object_fit="contain",
                    show_label=False,
                    allow_preview=True,
                )

        choices_state = gr.State(initial_choices)

        process_btn.click(
            fn=process_selected,
            inputs=[selected_images],
            outputs=[status_box, results_gallery],
        )

        def _refresh():
            gallery_data, checkbox_update, empty_results = refresh_gallery()
            choices = make_checkbox_choices(discover_raw_images())
            return gallery_data, checkbox_update, empty_results, choices

        refresh_btn.click(
            fn=_refresh,
            inputs=[],
            outputs=[raw_gallery, selected_images, results_gallery, choices_state],
        )

        select_all_btn.click(
            fn=lambda all_choices: all_choices,
            inputs=[choices_state],
            outputs=[selected_images],
        )

        deselect_btn.click(
            fn=lambda: [],
            inputs=[],
            outputs=[selected_images],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
