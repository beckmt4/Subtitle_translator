#!/usr/bin/env python
"""
Modified version of asr_translate_srt.py that keeps all intermediate files
"""

# Import the original script
import asr_translate_srt
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the directory of the input file from the command line
    input_arg = None
    for i, arg in enumerate(sys.argv):
        if i > 0 and not arg.startswith('-') and os.path.exists(arg):
            input_arg = arg
            break
    
    if input_arg:
        # Extract file info
        input_path = Path(input_arg)
        file_name = input_path.stem
        file_dir = input_path.parent
        
        # Print information about what files will be kept
        print(f"INFO: Will keep all intermediate files for {file_name}")
        print(f"Intermediate audio file will be saved as: {file_dir / f'{file_name}.asr.wav'}")
    
    # Call the original script's main function
    asr_translate_srt.main()