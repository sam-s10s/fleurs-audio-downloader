#!/usr/bin/env python3
"""
FLEURS Audio Dataset Downloader

Downloads audio samples from the FLEURS dataset for specified languages.
"""

import json
import tarfile
import hashlib
import random
from pathlib import Path
from typing import Dict, Any
import click
import soundfile as sf
import numpy as np
from tqdm import tqdm
import requests


# Language code mappings for FLEURS dataset
# Note: FLEURS only has en_us, not en_gb
LANGUAGE_CODES = {
    "en_us": "en_us",  # English (US) - only English variant available in FLEURS
    "fr_fr": "fr_fr",  # French
    "he_il": "he_il",  # Hebrew
    "hi_in": "hi_in",  # Hindi
    "th_th": "th_th",  # Thai
    "cmn_hans_cn": "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "de_de": "de_de",  # German
    "it_it": "it_it",  # Italian
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


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if it doesn't exist."""
    cache_dir = Path.cwd() / ".cache" / "fleurs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(language_code: str, split: str) -> str:
    """Generate a cache key for the archive."""
    content = f"{language_code}_{split}"
    return hashlib.md5(content.encode()).hexdigest()


def convert_to_pcm_mono(audio_data: np.ndarray, sample_rate: int, target_rate: int = 16000) -> tuple:
    """
    Convert audio to PCM 16000 16-bit mono format.
    
    Args:
        audio_data: Input audio array
        sample_rate: Original sample rate
        target_rate: Target sample rate (default: 16000)
    
    Returns:
        Tuple of (converted_audio, target_sample_rate)
    """
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if sample_rate != target_rate:
        # Simple resampling using scipy-style interpolation
        from scipy import signal
        num_samples = int(len(audio_data) * target_rate / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    # Ensure 16-bit range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    return audio_data.astype(np.float32), target_rate


def download_fleurs_metadata(language_code: str, split: str = "train") -> list:
    """
    Download metadata TSV file from FLEURS dataset.
    
    Args:
        language_code: FLEURS language code (e.g., 'en_us')
        split: Dataset split ('train', 'dev', 'test')
    
    Returns:
        List of metadata dictionaries
    """
    base_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
    tsv_url = f"{base_url}/{language_code}/{split}.tsv"
    
    try:
        response = requests.get(tsv_url)
        response.raise_for_status()
        
        # Parse TSV content (no header row)
        tsv_content = response.text.strip()
        lines = tsv_content.split('\n')
        
        metadata = []
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 6:  # Ensure we have all required fields
                metadata.append({
                    'id': int(parts[0]),
                    'file_name': parts[1],
                    'raw_transcription': parts[2],
                    'transcription': parts[3],
                    # Skip parts[4] which appears to be phonetic transcription
                    'num_samples': int(parts[5]),
                    'gender': parts[6] if len(parts) > 6 else 'OTHER'
                })
        
        return metadata
        
    except Exception as e:
        click.echo(f"Error downloading metadata for {language_code}: {e}", err=True)
        return []


def download_and_extract_tar(language_code: str, split: str) -> dict:
    """
    Download and extract the tar.gz archive for a language/split combination.
    Uses caching to avoid re-downloading the same archive.
    
    Args:
        language_code: FLEURS language code (e.g., 'en_us')
        split: Dataset split ('train', 'dev', 'test')
    
    Returns:
        Dictionary mapping file names to audio bytes
    """
    cache_dir = get_cache_dir()
    cache_key = get_cache_key(language_code, split)
    cache_file = cache_dir / f"{cache_key}_{language_code}_{split}.tar.gz"
    
    # Check if we have a cached version
    if cache_file.exists():
        click.echo(f"Using cached {split} archive for {language_code}")
        tar_file_path = str(cache_file)
    else:
        # Download the archive
        base_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
        tar_url = f"{base_url}/{language_code}/audio/{split}.tar.gz"
        
        click.echo(f"Downloading {split} archive for {language_code}...")
        
        try:
            response = requests.get(tar_url, stream=True)
            response.raise_for_status()
            
            # Download to cache
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            tar_file_path = str(cache_file)
            click.echo(f"Cached archive to {cache_file}")
            
        except Exception as e:
            raise Exception(f"Failed to download {split} archive for {language_code}: {e}")
    
    try:
        # Extract audio files from tar.gz
        audio_files = {}
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.wav'):
                    # Extract just the filename from the path
                    file_name = Path(member.name).name
                    # Extract the file content
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        audio_files[file_name] = file_obj.read()
        
        click.echo(f"Extracted {len(audio_files)} audio files from {split} archive")
        return audio_files
        
    except Exception as e:
        raise Exception(f"Failed to extract {split} archive for {language_code}: {e}")


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
        split: Dataset split to use ('train', 'dev', 'test')
    
    Returns:
        Dictionary with download statistics and metadata
    """
    click.echo(f"Loading {LANGUAGE_NAMES.get(language_code, language_code)} dataset from FLEURS...")
    
    try:
        # Download metadata for the language and split
        metadata_list = download_fleurs_metadata(language_code, split)
        
        if not metadata_list:
            return {
                "language_code": language_code,
                "error": "Failed to download metadata",
                "samples_downloaded": 0
            }
        
        # Randomly sample the requested number of samples
        if len(metadata_list) > num_samples:
            metadata_list = random.sample(metadata_list, num_samples)
            click.echo(f"Randomly selected {num_samples} samples from {len(metadata_list)} available")
        else:
            click.echo(f"Using all {len(metadata_list)} available samples")
        
        # Create language-specific output directory
        lang_dir = output_dir / LANGUAGE_NAMES.get(language_code, language_code).lower().replace(" ", "_")
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract the entire archive for this split
        try:
            audio_files = download_and_extract_tar(language_code, split)
        except Exception as e:
            return {
                "language_code": language_code,
                "error": f"Failed to download archive: {e}",
                "samples_downloaded": 0
            }
        
        # Process requested samples
        samples_downloaded = 0
        processed_metadata = []
        
        with tqdm(total=len(metadata_list), desc=f"Processing {LANGUAGE_NAMES.get(language_code, language_code)}") as pbar:
            for item in metadata_list:
                try:
                    # Check if we have this audio file
                    if item['file_name'] not in audio_files:
                        click.echo(f"Audio file {item['file_name']} not found in archive", err=True)
                        pbar.update(1)
                        continue
                    
                    # Get audio bytes
                    audio_bytes = audio_files[item['file_name']]
                    
                    # Generate local filename
                    audio_filename = f"{language_code}_{item['id']:06d}.wav"
                    audio_path = lang_dir / audio_filename
                    
                    # Save original audio file temporarily
                    temp_audio_path = audio_path.with_suffix('.temp.wav')
                    with open(temp_audio_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Load and convert audio to PCM 16000 16-bit mono
                    try:
                        audio_data, original_sample_rate = sf.read(str(temp_audio_path))
                        
                        # Convert to PCM mono format
                        converted_audio, target_sample_rate = convert_to_pcm_mono(audio_data, original_sample_rate)
                        
                        # Save converted audio
                        sf.write(str(audio_path), converted_audio, target_sample_rate, subtype='PCM_16')
                        
                        # Clean up temporary file
                        temp_audio_path.unlink()
                        
                        # Update metadata with converted values
                        duration_seconds = len(converted_audio) / target_sample_rate
                        sample_rate = target_sample_rate
                        num_samples = len(converted_audio)
                        
                    except Exception as e:
                        # Clean up temporary file if it exists
                        if temp_audio_path.exists():
                            temp_audio_path.unlink()
                        # Fallback if audio conversion fails
                        click.echo(f"Warning: Could not convert audio {item['file_name']}: {e}", err=True)
                        duration_seconds = item['num_samples'] / 16000  # Assume 16kHz
                        sample_rate = 16000
                        num_samples = item['num_samples']
                    
                    # Convert gender string to integer
                    gender_map = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
                    gender_int = gender_map.get(item['gender'], 2)
                    
                    # Store processed metadata
                    sample_metadata = {
                        "id": item['id'],
                        "filename": audio_filename,
                        "transcription": item['transcription'],
                        "raw_transcription": item['raw_transcription'],
                        "language": LANGUAGE_NAMES.get(language_code, language_code),
                        "gender": gender_int,
                        "num_samples": num_samples,
                        "sampling_rate": sample_rate,
                        "duration_seconds": duration_seconds,
                        "original_filename": item['file_name'],
                        "format": "PCM 16-bit mono"
                    }
                    processed_metadata.append(sample_metadata)
                    
                    samples_downloaded += 1
                    pbar.update(1)
                    
                except Exception as e:
                    click.echo(f"Failed to process {item['file_name']}: {e}", err=True)
                    pbar.update(1)
                    continue
        
        # Handle existing metadata (append mode)
        metadata_path = lang_dir / f"{language_code}_metadata.json"
        existing_metadata = []
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                click.echo(f"Found {len(existing_metadata)} existing samples, appending {len(processed_metadata)} new samples")
            except Exception as e:
                click.echo(f"Warning: Could not read existing metadata: {e}", err=True)
        
        # Combine existing and new metadata, avoiding duplicates by ID
        existing_ids = {item['id'] for item in existing_metadata}
        new_metadata = [item for item in processed_metadata if item['id'] not in existing_ids]
        
        if len(new_metadata) < len(processed_metadata):
            skipped = len(processed_metadata) - len(new_metadata)
            click.echo(f"Skipped {skipped} duplicate samples (already exist)")
        
        combined_metadata = existing_metadata + new_metadata
        
        # Save combined metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
        
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
    help="Dataset split to use (train, dev, test). Default: train"
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducible sampling. If not specified, uses random sampling."
)
@click.option(
    "--reset",
    is_flag=True,
    help="Clear output directory before downloading. Otherwise, append new samples to existing data."
)
@click.argument("output_dir", type=click.Path())
def cli(lang: tuple, samples: int, split: str, random_seed: int, reset: bool, output_dir: str):
    """
    Download audio samples from the FLEURS dataset.
    
    OUTPUT_DIR: Directory where audio files will be saved
    
    Examples:
        uv run fleurs-download --lang en_gb --lang fr_fr --samples 3 ./audio_samples
        uv run fleurs-download --lang hi_in --samples 5 --split validation ./data
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        click.echo(f"Using random seed: {random_seed}")
    
    # Convert output_dir to Path object
    output_path = Path(output_dir)
    
    # Handle reset option
    if reset and output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        click.echo(f"🗑️  Cleared output directory: {output_path}")
    
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
        # Validate and map language codes
        languages_to_download = []
        for lang_code in lang:
            if lang_code == "en_gb":
                click.echo("⚠️  Note: FLEURS dataset only has 'en_us' (English US), not 'en_gb' (English GB)")
                click.echo("    Using 'en_us' instead...")
                languages_to_download.append("en_us")
            elif lang_code in LANGUAGE_CODES:
                mapped_code = LANGUAGE_CODES[lang_code]
                languages_to_download.append(mapped_code)
            else:
                click.echo(f"❌ Unsupported language code: {lang_code}", err=True)
                click.echo(f"Supported codes: {', '.join(LANGUAGE_CODES.keys())}")
                return
    
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
