#!/usr/bin/env python3
import os
import sys
import shlex
import subprocess
from pathlib import Path
from utils.logger import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

ROLES = [
    "female_modern", "male_modern",
    "female_historical", "male_historical",
    # Add more roles as needed
]

# Provider configuration with multiple Groq models
PROVIDERS = {
    "groq": [
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        # Add more Groq models here
        # "mixtral-8x7b-32768",
    ]
}

def main():
    project_dir = Path(__file__).resolve().parent
    python_exe = sys.executable
    logging.info(f"Python: {python_exe}")
    logging.info(f"Project dir: {project_dir}")

    # Check API key availability
    api_key = os.getenv("GROQ_API_KEY")
    print(api_key)
    if not api_key:
        logging.error("Missing GROQ_API_KEY. Aborting.")
        sys.exit(1)

    failures = 0

    for role in ROLES:
        for model in PROVIDERS["groq"]:
            cmd = [
                python_exe, str(project_dir / "custom3.py"),
                f"--provider=groq",
                f"--model={model}",
                f"--role={role}",
                "--assessor=llama-3.1-8b-instant",
                # "--rag"  # Uncomment if needed
            ]

            logging.info(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(project_dir),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60 * 30  # 30 minutes
                )
            except subprocess.TimeoutExpired:
                failures += 1
                logging.error(f"[{role} | groq:{model}] ❌ TIMEOUT")
                continue
            except Exception as e:
                failures += 1
                logging.error(f"[{role} | groq:{model}] ❌ EXCEPTION: {str(e)}")
                continue

            if result.returncode != 0:
                failures += 1
                logging.error(f"[{role} | groq:{model}] ❌ FAILED (code {result.returncode})")
                if result.stderr:
                    logging.error(f"STDERR: {result.stderr.strip()}")
            else:
                logging.info(f"[{role} | groq:{model}] ✅ COMPLETED")
                if result.stdout:
                    logging.debug(f"STDOUT: {result.stdout.strip()}")
                if result.stderr:
                    logging.debug(f"STDERR: {result.stderr.strip()}")

    if failures:
        logging.error(f"Completed with {failures} failures")
        sys.exit(1)
    
    logging.info("All runs completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()