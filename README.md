# FLEURS Audio Dataset Downloader

A Python tool to download audio samples from the [FLEURS dataset](https://huggingface.co/datasets/google/fleurs) using UV for dependency management.

## Overview

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a speech dataset covering 102 languages with parallel sentences. This tool allows you to easily download audio samples for specific languages.

> **✅ Status**: This tool successfully downloads **real FLEURS audio data** directly from the Hugging Face dataset repository. It downloads the compressed archives, extracts the audio files, and provides them with complete metadata including transcriptions, speaker information, and audio characteristics.

## Supported Languages

The tool **dynamically discovers all available languages** from the FLEURS dataset on Hugging Face. Currently, this includes **49+ languages** such as:

Use `uv run fleurs-download --list` to see all currently available languages.

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for dependency management. Make sure you have UV installed:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Basic Usage

Download 3 samples for English to a specified directory:

```bash
uv run fleurs-download --lang en_us --samples 3 ./output_dir
```

### Multiple Languages

Download samples for multiple languages:

```bash
uv run fleurs-download --lang en_us --lang fr_fr --lang hi_in --samples 3 ./output_dir
```

### All Supported Languages

Download samples for all supported languages (if no `--lang` specified):

```bash
uv run fleurs-download --samples 3 ./output_dir
```

### Different Dataset Splits

Choose from different dataset splits:

```bash
# Use validation split
uv run fleurs-download --lang en_us --samples 5 --split validation ./output_dir

# Use test split
uv run fleurs-download --lang fr_fr --samples 2 --split test ./output_dir
```

## Help and Language Listing

```bash
# Show help (both work)
uv run fleurs-download -h
uv run fleurs-download --help

# List available languages
uv run fleurs-download --list
uv run fleurs-download -L
```

### Complete Help Output

```
Usage: fleurs-download [OPTIONS] [OUTPUT_DIR]

  Download audio samples from the FLEURS dataset.

  OUTPUT_DIR: Directory where audio files will be saved

  Examples:     uv run fleurs-download --lang en_gb --lang fr_fr --samples 3
  ./audio_samples     uv run fleurs-download --lang hi_in --samples 5 --split
  validation ./data

Options:
  -l, --lang TEXT            Language codes to download (e.g., en_us, fr_fr,
                             he_il). Can be specified multiple times.
  -s, --samples INTEGER      Number of samples to download per language
                             (default: 3)
  -p, --split TEXT           Dataset split to use (train, dev, test). Default:
                             train
  -r, --random-seed INTEGER  Random seed for reproducible sampling. If not
                             specified, uses random sampling.
  -R, --reset                Clear output directory before downloading.
                             Otherwise, append new samples to existing data.
  -n, --normalize            Normalize audio volume to -20dB RMS for
                             consistent loudness across samples.
  -L, --list                 List all available language codes and exit.
  -h, --help                 Show this message and exit.
```

### Available Languages Output

```
🔍 Fetching available languages from Hugging Face...
✅ Found 49 available languages from Hugging Face
📋 Available FLEURS language codes:
==================================================
  af_za           - Afrikaans (South Africa)
  am_et           - Amharic (Ethiopia)
  ar_eg           - Arabic (Egypt)
  as_in           - As (India)
  ast_es          - Ast (Spain)
  az_az           - Azerbaijani (Azerbaijan)
  be_by           - Belarusian (Belarus)
  bg_bg           - Bulgarian (Bulgaria)
  bn_in           - Bengali (India)
  bs_ba           - Bosnian (Bosnia and Herzegovina)
  ca_es           - Catalan (Spain)
  ceb_ph          - Ceb (Philippines)
  ckb_iq          - Ckb (Iraq)
  cmn_hans_cn     - Mandarin Chinese (Simplified)
  cs_cz           - Czech (Czech Republic)
  cy_gb           - Welsh (GB)
  da_dk           - Danish (Denmark)
  de_de           - German (Germany)
  el_gr           - El (Greece)
  en_us           - English (US)
  ... and 29 more languages

Total: 49 languages available

Usage examples:
  uv run fleurs-download -l en_us -s 3 ./output
  uv run fleurs-download -l fr_fr -l de_de -s 5 ./multi_lang
```

## Random Sampling

By default, the tool randomly selects samples from the available dataset:

```bash
# Random sampling (different samples each time)
uv run fleurs-download --lang en_us --samples 5 ./random_samples

# Reproducible sampling with seed
uv run fleurs-download --lang en_us --samples 5 --random-seed 42 ./reproducible_samples
```

This ensures you get a diverse set of samples rather than always the first N samples from the dataset.

## Volume Normalization

The `--normalize` option ensures consistent audio levels across all samples:

```bash
# Download with volume normalization
uv run fleurs-download -l en_us -s 5 -n ./normalized_audio

# Using shorthand options
uv run fleurs-download -l fr_fr -s 3 -n -r 42 ./data
```

**Benefits:**

- **Consistent Volume**: All samples normalized to -20dB RMS level
- **ML-Ready**: Uniform audio levels improve training consistency
- **Quality Control**: Prevents overly quiet or loud samples

## Reset vs Append Mode

By default, the tool appends new samples to existing data:

```bash
# First run - downloads 3 samples
uv run fleurs-download --lang en_us --samples 3 ./my_data

# Second run - adds 2 more samples (total: 5)
uv run fleurs-download --lang en_us --samples 2 ./my_data

# Reset mode - clears directory and downloads fresh
uv run fleurs-download --lang en_us --samples 5 --reset ./my_data
```

The tool automatically:

- **Skips duplicates**: Won't re-download samples with the same ID
- **Updates metadata**: Combines existing and new sample information
- **Preserves data**: Existing samples remain unless `--reset` is used

## Output Structure

The tool creates the following directory structure:

```
output_dir/
├── english/
│   ├── en_us_000001.wav
│   ├── en_us_000002.wav
│   ├── en_us_000003.wav
│   └── en_us_metadata.json
├── french/
│   ├── fr_fr_000001.wav
│   ├── fr_fr_000002.wav
│   ├── fr_fr_000003.wav
│   └── fr_fr_metadata.json
└── ...
```

Each language directory contains:

- **Audio files**: Real WAV files from FLEURS dataset, 16kHz sampling rate
- **Metadata file**: JSON file with actual transcriptions, speaker information, and audio metadata

## Real FLEURS Data

The tool downloads authentic FLEURS dataset content:

- **Audio**: Real speech recordings from native speakers, converted to PCM 16-bit mono 16000 Hz
- **Transcriptions**: Actual parallel sentences from the FLoRes benchmark
- **Metadata**: Complete information including speaker gender, audio duration, and sampling details
- **Quality**: Professional-grade speech data suitable for research and development
- **Caching**: Downloaded archives are cached in `.cache/` to avoid re-downloading
- **Format**: All audio files are standardised to PCM 16-bit mono 16000 Hz for consistency

## Metadata Format

The metadata JSON file contains detailed information about each audio sample:

```json
[
  {
    "id": 151,
    "filename": "en_us_000151.wav",
    "transcription": "sir richard branson's virgin group had a bid for the bank rejected prior to the bank's nationalisation",
    "raw_transcription": "Sir Richard Branson's Virgin Group had a bid for the bank rejected prior to the bank's nationalisation.",
    "language": "English",
    "gender": 1,
    "num_samples": 120000,
    "sampling_rate": 16000,
    "duration_seconds": 7.5,
    "original_filename": "11559549184357409250.wav",
    "format": "PCM 16-bit mono"
  }
]
```

## Examples

### Example 1: Download samples for specific languages

```bash
uv run fleurs-download --lang en_us --lang de_de --lang it_it --samples 5 ./my_audio_data
```

### Example 2: Download validation samples

```bash
uv run fleurs-download --lang hi_in --lang th_th --samples 10 --split validation ./validation_data
```

### Example 3: Download all languages with fewer samples

```bash
uv run fleurs-download --samples 1 ./quick_test
```

## Caching

The tool automatically caches downloaded archives in the `.cache/fleurs/` directory to improve performance:

- **First download**: Downloads and caches the archive (can be large, ~1-2GB per language)
- **Subsequent downloads**: Uses cached archive, much faster
- **Cache location**: `.cache/fleurs/` in the project directory
- **Cache management**: Archives are reused across different sample counts and output directories

To clear the cache:

```bash
rm -rf .cache/fleurs/
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

## Dataset Information

- **Source**: [Google FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- **Audio Format**: WAV, 16kHz sampling rate
- **Languages**: 102 languages total
- **Splits**: Train (~1000 samples), Validation (~400 samples), Test (~400 samples)

## License

This tool is provided as-is. Please refer to the [FLEURS dataset license](https://huggingface.co/datasets/google/fleurs) for usage terms of the downloaded data.
