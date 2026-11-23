"""Gradio UI builder for Hibiki-Zero interactive demo (hot-reload friendly)."""

import gradio as gr
from . import engine


def build_demo(bundle: engine.HibikiBundle) -> gr.Blocks:
    """Builds the Gradio Blocks application bound to an already-loaded HibikiBundle."""

    def _run(audio_in, temp_audio, temp_text, trailing_silence):
        yield from engine.stream_translate(
            bundle=bundle,
            audio_input=audio_in,
            temp_audio=temp_audio,
            temp_text=temp_text,
            trailing_silence_sec=trailing_silence,
        )

    with gr.Blocks(title="Hibiki-Zero 3B Demo") as demo:
        gr.Markdown(
            """
            # 🎙️ Hibiki-Zero (`kyutai/hibiki-zero-3b-pytorch-bf16`) — Speak, See & Listen
            Speak **French, Spanish, Portuguese, or German** into your microphone (or upload an audio file), then click **Translate Speech**.
            * **See:** Watch the English translation tokens stream live as the 12.5 Hz loop runs, plus the frame-aligned timeline plot.
            * **Listen:** Play the **Translated English Speech** (in your transferred voice) or the **Stereo Comparison** (Left ear = your original audio, Right ear = simultaneous translation).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Input Speech (French / Spanish / Portuguese / German)",
                )
                with gr.Accordion("Generation Settings", open=False):
                    temp_audio = gr.Slider(
                        0.0, 1.2, value=0.8, step=0.05, label="Audio Temperature (temp)"
                    )
                    temp_text = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.05,
                        label="Text Temperature (temp_text, 0 = greedy)",
                    )
                    trailing_silence = gr.Slider(
                        1.0,
                        5.0,
                        value=3.0,
                        step=0.5,
                        label="Trailing Silence Padding (sec) — lets buffered final words flush",
                    )
                run_btn = gr.Button("Translate Speech", variant="primary")

            with gr.Column(scale=1):
                text_out = gr.Textbox(
                    label="Streaming English Translation (Inner Monologue Text Tokens)",
                    lines=4,
                    placeholder="Translated English text will stream here...",
                )
                stats_out = gr.Textbox(label="Streaming Telemetry (12.5 Hz Step Loop)", lines=2)
                audio_out = gr.Audio(
                    label="Translated English Speech (24 kHz Mimi Output)", type="numpy"
                )
                stereo_out = gr.Audio(
                    label="Stereo Overlay (Left = Original Speech, Right = Simultaneous English Translation)",
                    type="numpy",
                )

        plot_out = gr.Plot(
            label="Simultaneous 12.5 Hz Timeline (Source vs. Translated Speech & Word Emissions)"
        )

        run_btn.click(
            fn=_run,
            inputs=[audio_in, temp_audio, temp_text, trailing_silence],
            outputs=[text_out, stats_out, audio_out, stereo_out, plot_out],
        )

    return demo


def launch_demo(bundle: engine.HibikiBundle, share: bool = True, inline: bool = True):
    """Closes any previously running Gradio server and launches a fresh instance."""
    gr.close_all()
    demo = build_demo(bundle)
    return demo.queue().launch(share=share, inline=inline, debug=False)
