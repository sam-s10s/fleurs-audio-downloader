#!/usr/bin/env python3
"""
FLEURS Audio Dataset Downloader

Downloads audio samples from the FLEURS dataset for specified languages.
"""

import hashlib
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Dict

import click
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm


def get_available_languages() -> dict:
    """
    Fetch available language codes from Hugging Face FLEURS dataset.

    Returns:
        Dictionary mapping language codes to friendly names
    """
    try:
        # Fetch the main data directory listing from Hugging Face
        base_url = "https://huggingface.co/datasets/google/fleurs/tree/main/data"
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()

        # Parse HTML to find language directories
        import re

        # Look for directory links in the format: href="/datasets/google/fleurs/tree/main/data/LANG_CODE"
        pattern = r'href="/datasets/google/fleurs/tree/main/data/([a-z_]+)"'
        matches = re.findall(pattern, response.text)

        # Filter out non-language directories (like README files, etc.)
        language_codes = []
        for match in matches:
            # Language codes typically have underscores and are 2-5 characters + underscore + 2-5 characters
            if "_" in match and len(match) >= 4 and len(match) <= 15:
                language_codes.append(match)

        # Create friendly names mapping
        language_names = {}
        for code in sorted(set(language_codes)):
            # Generate friendly names from language codes
            parts = code.split("_")
            if len(parts) >= 2:
                lang_part = parts[0]
                region_part = parts[1] if len(parts) > 1 else ""

                # Map common language codes to names
                lang_map = {
                    "en": "English",
                    "fr": "French",
                    "de": "German",
                    "es": "Spanish",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "ru": "Russian",
                    "zh": "Chinese",
                    "ja": "Japanese",
                    "ko": "Korean",
                    "ar": "Arabic",
                    "hi": "Hindi",
                    "th": "Thai",
                    "he": "Hebrew",
                    "tr": "Turkish",
                    "pl": "Polish",
                    "nl": "Dutch",
                    "sv": "Swedish",
                    "da": "Danish",
                    "no": "Norwegian",
                    "fi": "Finnish",
                    "cs": "Czech",
                    "hu": "Hungarian",
                    "ro": "Romanian",
                    "bg": "Bulgarian",
                    "hr": "Croatian",
                    "sk": "Slovak",
                    "sl": "Slovenian",
                    "et": "Estonian",
                    "lv": "Latvian",
                    "lt": "Lithuanian",
                    "mt": "Maltese",
                    "ga": "Irish",
                    "cy": "Welsh",
                    "eu": "Basque",
                    "ca": "Catalan",
                    "gl": "Galician",
                    "cmn": "Mandarin Chinese",
                    "yue": "Cantonese",
                    "vi": "Vietnamese",
                    "id": "Indonesian",
                    "ms": "Malay",
                    "tl": "Filipino",
                    "bn": "Bengali",
                    "ur": "Urdu",
                    "ta": "Tamil",
                    "te": "Telugu",
                    "ml": "Malayalam",
                    "kn": "Kannada",
                    "gu": "Gujarati",
                    "pa": "Punjabi",
                    "mr": "Marathi",
                    "ne": "Nepali",
                    "si": "Sinhala",
                    "my": "Burmese",
                    "km": "Khmer",
                    "lo": "Lao",
                    "ka": "Georgian",
                    "hy": "Armenian",
                    "az": "Azerbaijani",
                    "kk": "Kazakh",
                    "ky": "Kyrgyz",
                    "uz": "Uzbek",
                    "tg": "Tajik",
                    "mn": "Mongolian",
                    "fa": "Persian",
                    "ps": "Pashto",
                    "sw": "Swahili",
                    "am": "Amharic",
                    "om": "Oromo",
                    "so": "Somali",
                    "ha": "Hausa",
                    "yo": "Yoruba",
                    "ig": "Igbo",
                    "zu": "Zulu",
                    "xh": "Xhosa",
                    "af": "Afrikaans",
                    "is": "Icelandic",
                    "fo": "Faroese",
                    "mk": "Macedonian",
                    "sq": "Albanian",
                    "sr": "Serbian",
                    "bs": "Bosnian",
                    "me": "Montenegrin",
                    "be": "Belarusian",
                    "uk": "Ukrainian",
                }

                friendly_name = lang_map.get(lang_part, lang_part.title())
                if region_part:
                    region_map = {
                        "us": "US",
                        "gb": "GB",
                        "ca": "Canada",
                        "au": "Australia",
                        "in": "India",
                        "cn": "China",
                        "tw": "Taiwan",
                        "hk": "Hong Kong",
                        "sg": "Singapore",
                        "my": "Malaysia",
                        "id": "Indonesia",
                        "th": "Thailand",
                        "vn": "Vietnam",
                        "ph": "Philippines",
                        "kr": "Korea",
                        "jp": "Japan",
                        "il": "Israel",
                        "ae": "UAE",
                        "sa": "Saudi Arabia",
                        "eg": "Egypt",
                        "ma": "Morocco",
                        "dz": "Algeria",
                        "tn": "Tunisia",
                        "ly": "Libya",
                        "sd": "Sudan",
                        "et": "Ethiopia",
                        "ke": "Kenya",
                        "tz": "Tanzania",
                        "ug": "Uganda",
                        "rw": "Rwanda",
                        "mw": "Malawi",
                        "zm": "Zambia",
                        "zw": "Zimbabwe",
                        "bw": "Botswana",
                        "na": "Namibia",
                        "za": "South Africa",
                        "br": "Brazil",
                        "mx": "Mexico",
                        "ar": "Argentina",
                        "cl": "Chile",
                        "co": "Colombia",
                        "ve": "Venezuela",
                        "pe": "Peru",
                        "ec": "Ecuador",
                        "bo": "Bolivia",
                        "py": "Paraguay",
                        "uy": "Uruguay",
                        "gf": "French Guiana",
                        "sr": "Suriname",
                        "gy": "Guyana",
                        "fr": "France",
                        "be": "Belgium",
                        "ch": "Switzerland",
                        "lu": "Luxembourg",
                        "mc": "Monaco",
                        "de": "Germany",
                        "at": "Austria",
                        "it": "Italy",
                        "es": "Spain",
                        "pt": "Portugal",
                        "ad": "Andorra",
                        "sm": "San Marino",
                        "va": "Vatican",
                        "mt": "Malta",
                        "cy": "Cyprus",
                        "gr": "Greece",
                        "mk": "North Macedonia",
                        "al": "Albania",
                        "me": "Montenegro",
                        "rs": "Serbia",
                        "ba": "Bosnia and Herzegovina",
                        "hr": "Croatia",
                        "si": "Slovenia",
                        "hu": "Hungary",
                        "ro": "Romania",
                        "bg": "Bulgaria",
                        "md": "Moldova",
                        "ua": "Ukraine",
                        "by": "Belarus",
                        "ru": "Russia",
                        "ge": "Georgia",
                        "am": "Armenia",
                        "az": "Azerbaijan",
                        "kz": "Kazakhstan",
                        "kg": "Kyrgyzstan",
                        "uz": "Uzbekistan",
                        "tm": "Turkmenistan",
                        "tj": "Tajikistan",
                        "af": "Afghanistan",
                        "pk": "Pakistan",
                        "ir": "Iran",
                        "iq": "Iraq",
                        "sy": "Syria",
                        "lb": "Lebanon",
                        "jo": "Jordan",
                        "ps": "Palestine",
                        "tr": "Turkey",
                        "fi": "Finland",
                        "se": "Sweden",
                        "no": "Norway",
                        "dk": "Denmark",
                        "is": "Iceland",
                        "fo": "Faroe Islands",
                        "gl": "Greenland",
                        "ee": "Estonia",
                        "lv": "Latvia",
                        "lt": "Lithuania",
                        "pl": "Poland",
                        "cz": "Czech Republic",
                        "sk": "Slovakia",
                        "hans": "Simplified",
                        "hant": "Traditional",
                    }
                    region_friendly = region_map.get(region_part, region_part.upper())
                    friendly_name = f"{friendly_name} ({region_friendly})"

                language_names[code] = friendly_name

        click.echo(
            f"✅ Found {len(language_names)} available languages from Hugging Face"
        )
        return language_names

    except Exception as e:
        click.echo(f"⚠️  Could not fetch languages from Hugging Face: {e}", err=True)
        click.echo("Using fallback language list...", err=True)

        # Fallback to a basic set of known languages
        return {
            "en_us": "English (US)",
            "fr_fr": "French (France)",
            "de_de": "German (Germany)",
            "es_es": "Spanish (Spain)",
            "it_it": "Italian (Italy)",
            "pt_pt": "Portuguese (Portugal)",
            "ru_ru": "Russian (Russia)",
            "zh_cn": "Chinese (China)",
            "ja_jp": "Japanese (Japan)",
            "ko_kr": "Korean (Korea)",
            "ar_eg": "Arabic (Egypt)",
            "hi_in": "Hindi (India)",
            "th_th": "Thai (Thailand)",
            "he_il": "Hebrew (Israel)",
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


def normalize_audio_volume(
    audio_data: np.ndarray, target_db: float = -20.0
) -> np.ndarray:
    """
    Normalize audio volume to a target dB level.

    Args:
        audio_data: Input audio array
        target_db: Target RMS level in dB (default: -20 dB)

    Returns:
        Volume-normalized audio array
    """
    # Calculate RMS (Root Mean Square) of the audio
    rms = np.sqrt(np.mean(audio_data**2))

    # Avoid division by zero
    if rms == 0:
        return audio_data

    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20.0)

    # Calculate scaling factor
    scaling_factor = target_rms / rms

    # Apply scaling and clip to prevent clipping
    normalized_audio = audio_data * scaling_factor
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

    return normalized_audio


def convert_to_pcm_mono(
    audio_data: np.ndarray,
    sample_rate: int,
    target_rate: int = 16000,
    normalize: bool = False,
) -> tuple:
    """
    Convert audio to PCM 16000 16-bit mono format with optional volume normalization.

    Args:
        audio_data: Input audio array
        sample_rate: Original sample rate
        target_rate: Target sample rate (default: 16000)
        normalize: Whether to normalize volume (default: False)

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

    # Normalize volume if requested
    if normalize:
        audio_data = normalize_audio_volume(audio_data)

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
        lines = tsv_content.split("\n")

        metadata = []
        for line in lines:
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) >= 6:  # Ensure we have all required fields
                metadata.append(
                    {
                        "id": int(parts[0]),
                        "file_name": parts[1],
                        "raw_transcription": parts[2],
                        "transcription": parts[3],
                        # Skip parts[4] which appears to be phonetic transcription
                        "num_samples": int(parts[5]),
                        "gender": parts[6] if len(parts) > 6 else "OTHER",
                    }
                )

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
            with open(cache_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            tar_file_path = str(cache_file)
            click.echo(f"Cached archive to {cache_file}")

        except Exception as e:
            raise Exception(
                f"Failed to download {split} archive for {language_code}: {e}"
            )

    try:
        # Extract audio files from tar.gz
        audio_files = {}
        with tarfile.open(tar_file_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".wav"):
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
    split: str = "train",
    normalize: bool = False,
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
    click.echo(f"Loading {language_code} dataset from FLEURS...")

    try:
        # Download metadata for the language and split
        metadata_list = download_fleurs_metadata(language_code, split)

        if not metadata_list:
            return {
                "language_code": language_code,
                "error": "Failed to download metadata",
                "samples_downloaded": 0,
            }

        # Randomly sample the requested number of samples
        if len(metadata_list) > num_samples:
            metadata_list = random.sample(metadata_list, num_samples)
            click.echo(
                f"Randomly selected {num_samples} samples from {len(metadata_list)} available"
            )
        else:
            click.echo(f"Using all {len(metadata_list)} available samples")

        # Create language-specific output directory
        lang_dir = output_dir / language_code
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Download and extract the entire archive for this split
        try:
            audio_files = download_and_extract_tar(language_code, split)
        except Exception as e:
            return {
                "language_code": language_code,
                "error": f"Failed to download archive: {e}",
                "samples_downloaded": 0,
            }

        # Process requested samples
        samples_downloaded = 0
        processed_metadata = []

        with tqdm(total=len(metadata_list), desc=f"Processing {language_code}") as pbar:
            for item in metadata_list:
                try:
                    # Check if we have this audio file
                    if item["file_name"] not in audio_files:
                        click.echo(
                            f"Audio file {item['file_name']} not found in archive",
                            err=True,
                        )
                        pbar.update(1)
                        continue

                    # Get audio bytes
                    audio_bytes = audio_files[item["file_name"]]

                    # Generate local filename
                    audio_filename = f"{language_code}_{item['id']:06d}.wav"
                    audio_path = lang_dir / audio_filename

                    # Save original audio file temporarily
                    temp_audio_path = audio_path.with_suffix(".temp.wav")
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # Load and convert audio to PCM 16000 16-bit mono
                    try:
                        audio_data, original_sample_rate = sf.read(str(temp_audio_path))

                        # Convert to PCM mono format with optional normalization
                        converted_audio, target_sample_rate = convert_to_pcm_mono(
                            audio_data, original_sample_rate, normalize=normalize
                        )

                        # Save converted audio
                        sf.write(
                            str(audio_path),
                            converted_audio,
                            target_sample_rate,
                            subtype="PCM_16",
                        )

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
                        # Skip this sample if audio conversion fails
                        click.echo(
                            f"Error: Could not convert audio {item['file_name']}: {e}",
                            err=True,
                        )
                        pbar.update(1)
                        continue

                    # Convert gender string to integer
                    gender_map = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
                    gender_int = gender_map.get(item["gender"], 2)

                    # Store processed metadata
                    sample_metadata = {
                        "id": item["id"],
                        "filename": audio_filename,
                        "transcription": item["transcription"],
                        "raw_transcription": item["raw_transcription"],
                        "language": language_code,
                        "gender": gender_int,
                        "num_samples": num_samples,
                        "sampling_rate": sample_rate,
                        "duration_seconds": duration_seconds,
                        "original_filename": item["file_name"],
                        "format": "PCM 16-bit mono",
                        "normalized": normalize,
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
                with open(metadata_path, "r", encoding="utf-8") as f:
                    existing_metadata = json.load(f)
                click.echo(
                    f"Found {len(existing_metadata)} existing samples, processing {len(processed_metadata)} new samples"
                )
            except Exception as e:
                click.echo(f"Warning: Could not read existing metadata: {e}", err=True)

        # Combine existing and new metadata, overwriting duplicates by ID
        existing_ids = {item["id"]: item for item in existing_metadata}
        overwritten_count = 0

        # Update existing metadata with new samples, overwriting duplicates
        for new_item in processed_metadata:
            if new_item["id"] in existing_ids:
                overwritten_count += 1
            existing_ids[new_item["id"]] = new_item

        if overwritten_count > 0:
            click.echo(
                f"Overwritten {overwritten_count} existing samples with same IDs"
            )

        combined_metadata = list(existing_ids.values())

        # Save combined metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2, ensure_ascii=False)

        return {
            "language_code": language_code,
            "samples_downloaded": samples_downloaded,
            "output_directory": str(lang_dir),
            "metadata_file": str(metadata_path),
        }

    except Exception as e:
        click.echo(f"Error downloading {language_code}: {str(e)}", err=True)
        return {
            "language_code": language_code,
            "error": str(e),
            "samples_downloaded": 0,
        }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--lang",
    "-l",
    multiple=True,
    help="Language codes to download (e.g., en_us, fr_fr, he_il). Can be specified multiple times.",
)
@click.option(
    "--samples",
    "-s",
    default=3,
    help="Number of samples to download per language (default: 3)",
)
@click.option(
    "--split",
    "-p",
    default="train",
    help="Dataset split to use (train, dev, test). Default: train",
)
@click.option(
    "--random-seed",
    "-r",
    type=int,
    help="Random seed for reproducible sampling. If not specified, uses random sampling.",
)
@click.option(
    "--reset",
    "-R",
    is_flag=True,
    help="Clear output directory before downloading. Otherwise, append new samples to existing data.",
)
@click.option(
    "--normalize",
    "-n",
    is_flag=True,
    help="Normalize audio volume to -20dB RMS for consistent loudness across samples.",
)
@click.option(
    "--list", "-L", is_flag=True, help="List all available language codes and exit."
)
@click.argument("output_dir", type=click.Path(), required=False)
def cli(
    lang: tuple,
    samples: int,
    split: str,
    random_seed: int,
    reset: bool,
    normalize: bool,
    list: bool,
    output_dir: str,
):
    """
    Download audio samples from the FLEURS dataset.

    OUTPUT_DIR: Directory where audio files will be saved

    Examples:
        uv run fleurs-download --lang en_us --lang fr_fr --samples 3 ./audio_samples
        uv run fleurs-download --lang hi_in --samples 5 --split validation ./data
    """
    # Handle --list option
    if list:
        click.echo("🔍 Fetching available languages from Hugging Face...")
        available_languages = get_available_languages()

        click.echo("📋 Available FLEURS language codes:")
        click.echo("=" * 50)
        for lang_code, lang_name in available_languages.items():
            click.echo(f"  {lang_code:<15} - {lang_name}")
        click.echo(f"\nTotal: {len(available_languages)} languages available")
        click.echo("\nUsage examples:")
        click.echo("  uv run fleurs-download -l en_us -s 3 ./output")
        click.echo("  uv run fleurs-download -l fr_fr -l de_de -s 5 ./multi_lang")
        return

    # Require output_dir if not listing
    if not output_dir:
        click.echo("Error: Missing argument 'OUTPUT_DIR'.", err=True)
        click.echo(
            "Use --help for usage information or --list to see available languages."
        )
        return

    # Set random seed if provided, or generate one for display
    if random_seed is not None:
        random.seed(random_seed)
        actual_seed = random_seed
        click.echo(f"🎲 Using random seed: {random_seed}")
    else:
        # Generate a random seed for reproducibility tracking
        import time

        actual_seed = int(time.time() * 1000000) % 2147483647  # Keep within int32 range
        random.seed(actual_seed)
        click.echo(
            f"🎲 Generated random seed: {actual_seed} (use --random-seed {actual_seed} to reproduce)"
        )

    # Show normalization status
    if normalize:
        click.echo("🔊 Volume normalization enabled (-20dB RMS)")

    # Fetch available languages from Hugging Face
    click.echo("🔍 Fetching available languages from Hugging Face...")
    available_languages = get_available_languages()

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
        languages_to_download = [k for k in available_languages.keys()]
        click.echo(
            f"No languages specified. Downloading all {len(languages_to_download)} supported languages:"
        )
        for lang_code in languages_to_download[:5]:  # Show first 5
            click.echo(
                f"  - {available_languages.get(lang_code, lang_code)} ({lang_code})"
            )
        if len(languages_to_download) > 5:
            click.echo(f"  ... and {len(languages_to_download) - 5} more")
    else:
        # Validate language codes
        languages_to_download = []
        invalid_languages = []
        
        for lang_code in lang:
            if lang_code in available_languages:
                languages_to_download.append(lang_code)
            else:
                invalid_languages.append(lang_code)
        
        if invalid_languages:
            click.echo(f"❌ Invalid language codes: {', '.join(invalid_languages)}", err=True)
            available_codes = [k for k in available_languages.keys()][:10]
            click.echo(f"Available codes: {', '.join(available_codes)}...")
            click.echo("Use --list to see all available languages")
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
            language_code, samples, output_path, split, normalize
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
            samples_count = result["samples_downloaded"]
            total_samples += samples_count
            click.echo(f"✅ {result['language_code']}: {samples_count} samples")
            click.echo(f"   📁 {result['output_directory']}")

    click.echo(f"\nTotal samples downloaded: {total_samples}")
    click.echo(f"Output directory: {output_path}")


if __name__ == "__main__":
    cli()
