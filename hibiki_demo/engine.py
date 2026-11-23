"""Core 12.5 Hz streaming translation engine for kyutai/hibiki-zero-3b-pytorch-bf16."""

from dataclasses import dataclass
import gc
import time
from typing import Any, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from moshi.models import LMGen, loaders

DEFAULT_HF_REPO = "kyutai/hibiki-zero-3b-pytorch-bf16"


@dataclass
class HibikiBundle:
    """Holds loaded GPU models in Colab VRAM across hot-reloads of engine.py and ui.py."""
    hf_repo: str
    checkpoint_info: Any
    mimi: Any
    text_tokenizer: Any
    lm_model: Any
    device: torch.device
    dtype: torch.dtype
    sample_rate: int
    frame_rate: float
    frame_size: int


def detect_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Auto-selects CUDA + bfloat16 (Ampere/Ada/Hopper SM>=8.0) or float16 (Turing T4 SM 7.5)."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32
    device = torch.device("cuda")
    cc_major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
    return device, dtype


def load_models(
    hf_repo: str = DEFAULT_HF_REPO,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> HibikiBundle:
    """
    Loads Mimi, SentencePiece tokenizer, and Hibiki-Zero 3B once into GPU memory.
    Keep the returned HibikiBundle in a notebook variable so subsequent code reloads
    do not need to re-download or reload the 3B weights.
    """
    auto_device, auto_dtype = detect_device_and_dtype()
    device = device or auto_device
    dtype = dtype or auto_dtype

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | Precision: {dtype}")
    else:
        print(f"Running on CPU | Precision: {dtype}")

    print(f"Fetching checkpoint metadata from {hf_repo}...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo)

    print("Loading Mimi streaming neural audio codec...")
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.eval()

    print("Loading SentencePiece text tokenizer...")
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    print(f"Loading Hibiki-Zero 3B language model on {device} ({dtype})...")
    lm_model = checkpoint_info.get_moshi(device=device, dtype=dtype)
    lm_model.eval()

    sample_rate = int(mimi.sample_rate)       # 24000 Hz
    frame_rate = float(mimi.frame_rate)       # 12.5 Hz (80 ms per step)
    frame_size = int(sample_rate / frame_rate)  # 1920 samples per 80 ms frame

    print(
        f"Ready! Sample rate: {sample_rate} Hz | "
        f"Frame rate: {frame_rate} Hz ({frame_size} samples / 80 ms frame)"
    )
    return HibikiBundle(
        hf_repo=hf_repo,
        checkpoint_info=checkpoint_info,
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm_model=lm_model,
        device=device,
        dtype=dtype,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        frame_size=frame_size,
    )


def preprocess_audio(
    bundle: HibikiBundle,
    audio_tuple: Tuple[int, np.ndarray],
    trailing_silence_sec: float = 3.0,
) -> np.ndarray:
    """
    Converts Gradio (sr, numpy_array) audio to 24 kHz mono float32
    and appends trailing silence so buffered simultaneous words flush out.
    """
    sr, wav = audio_tuple
    if wav.dtype.kind in ("i", "u"):
        max_val = float(np.iinfo(wav.dtype).max)
        wav = wav.astype(np.float32) / max_val
    else:
        wav = wav.astype(np.float32)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)

    if sr != bundle.sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=bundle.sample_rate)

    peak = np.max(np.abs(wav))
    if peak > 1e-4:
        wav = wav / max(peak, 1.0) * 0.95

    silence_samples = int(trailing_silence_sec * bundle.sample_rate)
    wav_padded = np.concatenate([wav, np.zeros(silence_samples, dtype=np.float32)])

    rem = len(wav_padded) % bundle.frame_size
    if rem != 0:
        wav_padded = np.concatenate(
            [wav_padded, np.zeros(bundle.frame_size - rem, dtype=np.float32)]
        )

    return wav_padded


def build_alignment_figure(
    src_wav: np.ndarray,
    trg_wav: np.ndarray,
    token_events: list,
    total_duration_sec: float,
):
    """Plots source waveform, translated waveform, and emitted English word timestamps."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    t_axis = np.linspace(0, total_duration_sec, len(src_wav))

    ax1.plot(t_axis, src_wav, color="#3b82f6", linewidth=0.7, alpha=0.85)
    ax1.set_ylabel("Source Speech")
    ax1.set_title(
        "Simultaneous 12.5 Hz Timeline (Source Input vs. Translated English Audio & Text Tokens)"
    )
    ax1.grid(True, alpha=0.25)

    ax2.plot(t_axis, trg_wav, color="#10b981", linewidth=0.7, alpha=0.85)
    ax2.set_ylabel("Translated Audio")
    ax2.set_xlabel("Model Time (seconds @ 12.5 Hz)")
    ax2.grid(True, alpha=0.25)

    for idx, (t_sec, piece) in enumerate(token_events):
        clean = piece.strip()
        if not clean:
            continue
        ax2.axvline(t_sec, color="#f59e0b", linestyle="--", linewidth=0.6, alpha=0.5)
        y_pos = 0.75 if (idx % 2 == 0) else -0.82
        ax2.text(
            t_sec,
            y_pos,
            clean,
            fontsize=8,
            rotation=25,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fef3c7", ec="#f59e0b", alpha=0.85),
        )

    ax1.set_ylim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def stream_translate(
    bundle: HibikiBundle,
    audio_input: Optional[Tuple[int, np.ndarray]],
    temp_audio: float = 0.8,
    temp_text: float = 0.0,
    trailing_silence_sec: float = 3.0,
):
    """
    Generator for Gradio that steps Hibiki-Zero frame-by-frame (80 ms per step),
    yielding live text updates as translation progresses, followed by the final
    translated audio, stereo comparison track, and timeline plot.
    """
    if audio_input is None:
        yield (
            "Please record from your microphone or upload an audio clip first.",
            "",
            None,
            None,
            None,
        )
        return

    src_wav = preprocess_audio(bundle, audio_input, trailing_silence_sec=trailing_silence_sec)
    num_frames = len(src_wav) // bundle.frame_size
    total_duration_sec = len(src_wav) / bundle.sample_rate

    gen_kwargs = dict(getattr(bundle.checkpoint_info, "lm_gen_config", {}) or {})
    gen_kwargs["temp"] = float(temp_audio)
    gen_kwargs["temp_text"] = float(temp_text)
    lm_gen = LMGen(bundle.lm_model, **gen_kwargs)

    emitted_pieces = []
    token_events = []
    decoded_pcm_chunks = []
    wait_frames = 0

    t_start = time.perf_counter()

    with torch.no_grad(), bundle.mimi.streaming(1), lm_gen.streaming(1):
        for frame_idx in range(num_frames):
            frame_time_sec = frame_idx * (bundle.frame_size / bundle.sample_rate)
            chunk_np = src_wav[
                frame_idx * bundle.frame_size : (frame_idx + 1) * bundle.frame_size
            ]
            chunk_tensor = (
                torch.from_numpy(chunk_np)
                .to(device=bundle.device, dtype=torch.float32)
                .view(1, 1, bundle.frame_size)
            )

            # 1. Encode 80 ms source audio frame into Mimi acoustic tokens
            codes = bundle.mimi.encode(chunk_tensor)

            # 2. Step the Hibiki-Zero language model by one 80 ms frame
            tokens = lm_gen.step(codes)
            if tokens is None:
                decoded_pcm_chunks.append(np.zeros(bundle.frame_size, dtype=np.float32))
                continue

            # 3. Decode target English speech tokens (channels 1..Q) back to 24 kHz PCM
            audio_tokens = tokens[:, 1:]
            pcm_out = bundle.mimi.decode(audio_tokens)
            pcm_np = pcm_out[0, 0].detach().cpu().numpy()
            decoded_pcm_chunks.append(pcm_np)

            # 4. Decode inner-monologue English text token (channel 0)
            text_token_id = int(tokens[0, 0, 0].item())
            if text_token_id not in (0, 3):  # 0 and 3 are special PAD / EPAD control tokens
                piece = bundle.text_tokenizer.id_to_piece(text_token_id).replace("▁", " ")
                emitted_pieces.append(piece)
                token_events.append((frame_time_sec, piece))

                live_transcript = "".join(emitted_pieces).strip()
                elapsed = time.perf_counter() - t_start
                rtf = elapsed / max(frame_time_sec, 0.08)
                status = (
                    f"Streaming frame {frame_idx + 1}/{num_frames} "
                    f"({frame_time_sec:.2f}s / {total_duration_sec:.2f}s) | RTF: {rtf:.2f}x"
                )
                yield live_transcript, status, None, None, None
            else:
                wait_frames += 1

    elapsed_total = time.perf_counter() - t_start
    rtf_total = elapsed_total / max(total_duration_sec, 0.08)
    trg_wav = np.concatenate(decoded_pcm_chunks)

    # Build a stereo comparison track: Left = original speech, Right = simultaneous translation
    stereo_wav = np.stack([src_wav[: len(trg_wav)], trg_wav], axis=-1)

    final_transcript = (
        "".join(emitted_pieces).strip()
        or "(No English text tokens emitted — try speaking louder or longer.)"
    )
    first_word_lag = f"{token_events[0][0]:.2f}s" if token_events else "N/A"
    summary_stats = (
        f"Done in {elapsed_total:.2f}s for {total_duration_sec:.2f}s audio "
        f"(RTF: {rtf_total:.2f}x — <1.0x is faster than real-time)\n"
        f"First token emitted at: {first_word_lag} | Emitted text tokens: {len(token_events)} | "
        f"Wait (PAD/EPAD) frames: {wait_frames}/{num_frames}"
    )

    fig = build_alignment_figure(
        src_wav[: len(trg_wav)], trg_wav, token_events, total_duration_sec
    )

    yield (
        final_transcript,
        summary_stats,
        (bundle.sample_rate, trg_wav),
        (bundle.sample_rate, stereo_wav),
        fig,
    )
