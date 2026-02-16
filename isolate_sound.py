#!/usr/bin/env python3
"""
Audio segmentation script using SAM-Audio.
Isolates specific sounds from an audio file based on text descriptions.
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from sam_audio import SAMAudio, SAMAudioProcessor


def isolate_sound(
    input_audio: str,
    sound_description: str,
    model_size: str = "large",
    output_dir: str = "output",
    predict_spans: bool = False,
    reranking_candidates: int = 1,
):
    """
    Isolate a specific sound from an audio file using SAM-Audio.

    Args:
        input_audio: Path to input audio file
        sound_description: Text description of the sound to isolate (e.g., "man speaking", "dog barking")
        model_size: Model size to use ("small" or "large")
        output_dir: Directory to save output files
        predict_spans: Whether to use span prediction for better accuracy (slower)
        reranking_candidates: Number of candidates for reranking (higher = better but slower)
    """
    print(f"Loading SAM-Audio model ({model_size})...")
    model_name = f"facebook/sam-audio-{model_size}"

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires a GPU to run. "
            "Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support installed."
        )

    # Load model and processor
    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)

    # Move model to GPU
    model = model.eval().cuda()
    print("Model loaded on GPU")

    # Verify input file exists
    input_path = Path(input_audio)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_audio}")

    print(f"Processing audio: {input_audio}")
    print(f"Target sound: {sound_description}")

    # Prepare input batch
    batch = processor(
        audios=[str(input_path)],
        descriptions=[sound_description.lower()],  # Use lowercase as recommended
    ).to("cuda")

    # Run separation
    print("Running audio separation...")
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get sample rate from processor
    sample_rate = processor.audio_sampling_rate

    # Generate output filenames
    input_stem = input_path.stem
    # Sanitize sound description for filename (replace spaces and special chars with underscore)
    safe_description = sound_description.lower().replace(" ", "_").replace("-", "_")
    safe_description = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_description)
    target_file = output_path / f"{input_stem}_{safe_description}_target.wav"
    residual_file = output_path / f"{input_stem}_{safe_description}_residual.wav"

    # Save results
    print(f"Saving isolated sound to: {target_file}")
    torchaudio.save(str(target_file), result.target[0].cpu(), sample_rate)

    print(f"Saving residual audio to: {residual_file}")
    torchaudio.save(str(residual_file), result.residual[0].cpu(), sample_rate)

    print("✓ Audio separation complete!")
    return target_file, residual_file


def main():
    parser = argparse.ArgumentParser(
        description="Isolate specific sounds from audio files using SAM-Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python isolate_sound.py audio.wav "dog barking"
  python isolate_sound.py audio.wav "man speaking" --model-size small
  python isolate_sound.py audio.wav "car engine" --output-dir results --predict-spans
        """
    )

    parser.add_argument(
        "input_audio",
        type=str,
        help="Path to input audio file"
    )

    parser.add_argument(
        "sound_description",
        type=str,
        help="Text description of the sound to isolate (e.g., 'man speaking', 'dog barking')"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "large"],
        default="large",
        help="SAM-Audio model size to use (default: large)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files (default: output)"
    )

    parser.add_argument(
        "--predict-spans",
        action="store_true",
        help="Enable span prediction for better accuracy (increases latency and memory usage)"
    )

    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Number of reranking candidates (higher = better but slower, default: 1)"
    )

    args = parser.parse_args()

    try:
        isolate_sound(
            input_audio=args.input_audio,
            sound_description=args.sound_description,
            model_size=args.model_size,
            output_dir=args.output_dir,
            predict_spans=args.predict_spans,
            reranking_candidates=args.reranking_candidates,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
