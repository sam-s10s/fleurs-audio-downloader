#!/usr/bin/env python3
"""
FLEURS Audio Dataset Downloader

Downloads audio samples from the FLEURS dataset for specified languages.
"""

import json
from pathlib import Path
from typing import Dict, Any
import click
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm


# Language code mappings for FLEURS dataset
LANGUAGE_CODES = {
    "en_gb": "en_us",  # English (using US variant as GB not available)
    "en_us": "en_us",
    "fr_fr": "fr_fr",
    "he_il": "he_il", 
    "hi_in": "hi_in",  # Hindi (Indian)
    "th_th": "th_th",
    "cmn_hans_cn": "cmn_hans_cn",  # Mandarin Chinese
    "de_de": "de_de",
    "it_it": "it_it",
}

# Friendly language names
LANGUAGE_NAMES = {
    "en_us": "English",
    "fr_fr": "French", 
    "he_il": "Hebrew",
    "hi_in": "Hindi",
    "th_th": "Thai",
    "cmn_hans_cn": "Mandarin Chinese",
    "de_de": "German",
    "it_it": "Italian",
}


def download_language_samples(
    language_code: str, 
    num_samples: int, 
    output_dir: Path,
    split: str = "train"
) -> Dict[str, Any]:
    """
    Download audio samples for a specific language from FLEURS dataset.
    
    Args:
        language_code: FLEURS language code (e.g., 'en_us')
        num_samples: Number of samples to download
        output_dir: Directory to save the audio files
        split: Dataset split to use ('train', 'validation', 'test')
    
    Returns:
        Dictionary with download statistics and metadata
    """
    click.echo(f"Loading {LANGUAGE_NAMES.get(language_code, language_code)} dataset...")
    
    try:
        # Load the dataset for the specific language
        dataset = load_dataset("google/fleurs", language_code, split=split, streaming=True)
        
        # Create language-specific output directory
        lang_dir = output_dir / LANGUAGE_NAMES.get(language_code, language_code).lower().replace(" ", "_")
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Download samples with progress bar
        samples_downloaded = 0
        metadata = []
        
        with tqdm(total=num_samples, desc=f"Downloading {LANGUAGE_NAMES.get(language_code, language_code)}") as pbar:
            for i, sample in enumerate(dataset):
                if samples_downloaded >= num_samples:
                    break
                
                # Generate filename
                audio_filename = f"{language_code}_{sample['id']:06d}.wav"
                audio_path = lang_dir / audio_filename
                
                # Save audio file
                audio_array = sample['audio']['array']
                sampling_rate = sample['audio']['sampling_rate']
                sf.write(str(audio_path), audio_array, sampling_rate)
                
                # Store metadata
                sample_metadata = {
                    "id": sample['id'],
                    "filename": audio_filename,
                    "transcription": sample['transcription'],
                    "raw_transcription": sample['raw_transcription'],
                    "language": sample['language'],
                    "gender": sample['gender'],
                    "num_samples": sample['num_samples'],
                    "sampling_rate": sampling_rate,
                    "duration_seconds": len(audio_array) / sampling_rate
                }
                metadata.append(sample_metadata)
                
                samples_downloaded += 1
                pbar.update(1)
        
        # Save metadata to JSON file
        metadata_path = lang_dir / f"{language_code}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {
            "language_code": language_code,
            "language_name": LANGUAGE_NAMES.get(language_code, language_code),
            "samples_downloaded": samples_downloaded,
            "output_directory": str(lang_dir),
            "metadata_file": str(metadata_path)
        }
        
    except Exception as e:
        click.echo(f"Error downloading {language_code}: {str(e)}", err=True)
        return {
            "language_code": language_code,
            "error": str(e),
            "samples_downloaded": 0
        }


@click.command()
@click.option(
    "--lang", 
    multiple=True,
    help="Language codes to download (e.g., en_gb, fr_fr, he_il). Can be specified multiple times."
)
@click.option(
    "--samples", 
    default=3, 
    help="Number of samples to download per language (default: 3)"
)
@click.option(
    "--split",
    default="train",
    type=click.Choice(["train", "validation", "test"]),
    help="Dataset split to use (default: train)"
)
@click.argument("output_dir", type=click.Path())
def cli(lang: tuple, samples: int, split: str, output_dir: str):
    """
    Download audio samples from the FLEURS dataset.
    
    OUTPUT_DIR: Directory where audio files will be saved
    
    Examples:
        uv run fleurs-download --lang en_gb --lang fr_fr --samples 3 ./audio_samples
        uv run fleurs-download --lang hi_in --samples 5 --split validation ./data
    """
    # Convert output_dir to Path object
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If no languages specified, use all supported languages
    if not lang:
        languages_to_download = [
            "en_us", "fr_fr", "he_il", "hi_in", 
            "th_th", "cmn_hans_cn", "de_de", "it_it"
        ]
        click.echo("No languages specified. Downloading all supported languages:")
        for lang_code in languages_to_download:
            click.echo(f"  - {LANGUAGE_NAMES.get(lang_code, lang_code)} ({lang_code})")
    else:
        # Map input language codes to FLEURS codes
        languages_to_download = []
        for input_lang in lang:
            fleurs_code = LANGUAGE_CODES.get(input_lang)
            if fleurs_code:
                languages_to_download.append(fleurs_code)
            else:
                click.echo(f"Warning: Language code '{input_lang}' not supported. Skipping.", err=True)
                click.echo(f"Supported codes: {', '.join(LANGUAGE_CODES.keys())}")
    
    if not languages_to_download:
        click.echo("No valid languages to download. Exiting.", err=True)
        return
    
    click.echo(f"\nDownloading {samples} samples per language to: {output_path}")
    click.echo(f"Using dataset split: {split}")
    click.echo("-" * 50)
    
    # Download samples for each language
    results = []
    for language_code in languages_to_download:
        result = download_language_samples(
            language_code=language_code,
            num_samples=samples,
            output_dir=output_path,
            split=split
        )
        results.append(result)
    
    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("DOWNLOAD SUMMARY")
    click.echo("=" * 50)
    
    total_samples = 0
    for result in results:
        if "error" in result:
            click.echo(f"❌ {result['language_code']}: ERROR - {result['error']}")
        else:
            samples_count = result['samples_downloaded']
            total_samples += samples_count
            click.echo(f"✅ {result['language_name']} ({result['language_code']}): {samples_count} samples")
            click.echo(f"   📁 {result['output_directory']}")
    
    click.echo(f"\nTotal samples downloaded: {total_samples}")
    click.echo(f"Output directory: {output_path}")


if __name__ == "__main__":
    cli()
