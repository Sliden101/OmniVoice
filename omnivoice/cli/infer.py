"""Single-item inference CLI for OmniVoice.

Generates audio from a single text input using voice cloning,
voice design, or auto voice.

Usage:
    # Voice cloning
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

    # Voice design
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --instruct "male, British accent" --output out.wav

    # Auto voice
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." --output out.wav
"""

import argparse
import logging

import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import str2bool


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--early_termination",
        type=str2bool,
        default=False,
        help="Stop generation for converged samples early.",
    )
    parser.add_argument(
        "--use_credit_decoding",
        type=str2bool,
        default=False,
        help="Use credit decoding to commit stable tokens earlier.",
    )
    parser.add_argument(
        "--use_torch_compile",
        type=str2bool,
        default=False,
        help="Use torch.compile for faster forward pass.",
    )
    parser.add_argument(
        "--fast_mode",
        type=str2bool,
        default=False,
        help="Enable all optimizations and halve num_step.",
    )
    parser.add_argument(
        "--ultra_mode",
        type=str2bool,
        default=False,
        help="Ultra-fast mode: 1/4 steps + all optimizations + boosted credit.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4"],
        help="Weight quantization for reduced memory usage.",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="Path to smaller draft model for speculative decoding.",
    )
    parser.add_argument(
        "--use_kv_cache",
        type=str2bool,
        default=False,
        help="Enable Window-Diffusion KV cache for faster inference.",
    )
    parser.add_argument(
        "--kv_external_window",
        type=int,
        default=128,
        help="External window length for KV cache.",
    )
    parser.add_argument(
        "--kv_internal_window",
        type=int,
        default=16,
        help="Internal window length for KV cache (active tokens).",
    )
    parser.add_argument(
        "--kv_refresh_cycle",
        type=int,
        default=4,
        help="KV cache refresh interval (full forward pass every N steps).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16,
        quantization=args.quantization,
    )

    if args.draft_model is not None:
        model.set_draft_model(args.draft_model)

    logging.info(f"Generating audio for: {args.text[:80]}...")
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        early_termination=args.early_termination,
        use_credit_decoding=args.use_credit_decoding,
        use_torch_compile=args.use_torch_compile,
        fast_mode=args.fast_mode,
        ultra_mode=args.ultra_mode,
        use_kv_cache=args.use_kv_cache,
        kv_external_window=args.kv_external_window,
        kv_internal_window=args.kv_internal_window,
        kv_refresh_cycle=args.kv_refresh_cycle,
    )

    torchaudio.save(args.output, audios[0], model.sampling_rate)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
