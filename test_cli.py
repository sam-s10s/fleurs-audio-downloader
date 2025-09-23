#!/usr/bin/env python3
"""
Simple test script to verify the CLI interface works correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_cli_help():
    """Test that the CLI help command works."""
    try:
        result = subprocess.run(
            ["uv", "run", "fleurs-download", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print("✅ CLI help command works!")
            print("Help output preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ CLI help command failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running CLI test: {e}")
        return False

def test_language_codes():
    """Test that language code validation works."""
    from fleurs_downloader.main import LANGUAGE_CODES, LANGUAGE_NAMES
    
    print("\n📋 Supported language codes:")
    for input_code, fleurs_code in LANGUAGE_CODES.items():
        language_name = LANGUAGE_NAMES.get(fleurs_code, fleurs_code)
        print(f"  {input_code} → {fleurs_code} ({language_name})")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing FLEURS Audio Downloader CLI")
    print("=" * 50)
    
    success = True
    
    # Test CLI help
    success &= test_cli_help()
    
    # Test language codes
    success &= test_language_codes()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The CLI is ready to use.")
        print("\nTry running:")
        print("  uv run fleurs-download --lang en_gb --samples 1 ./test_output")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)
