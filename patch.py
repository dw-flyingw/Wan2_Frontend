#!/usr/bin/env python3
"""
Patch script for Wan2_Frontend.

This script patches the upstream Wan2.2 repository with modifications
required for this frontend to function properly.

Patches:
- generate.py: Adds OpenAI prompt extension support
- wan/utils/prompt_extend.py: Adds OpenAIPromptExpander class
- wan/utils/system_prompt.py: Adds animate-specific system prompts

Usage:
    python patch.py patch     # Apply patches to Wan2.2 repo
    python patch.py restore   # Restore original files from backups
    python patch.py status    # Check patch status
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PATCHES_DIR = SCRIPT_DIR / "patches"
WAN2_2_REPO = Path(os.environ.get("WAN2_2_REPO", SCRIPT_DIR.parent / "Wan2.2"))
BACKUP_SUFFIX = ".orig"

# Files to patch (relative to their respective root directories)
PATCH_FILES = [
    "generate.py",
    "wan/utils/prompt_extend.py",
    "wan/utils/system_prompt.py",
]


def get_file_paths(filename: str) -> tuple[Path, Path, Path]:
    """Get source, destination, and backup paths for a file."""
    source = PATCHES_DIR / filename
    dest = WAN2_2_REPO / filename
    backup = WAN2_2_REPO / (filename + BACKUP_SUFFIX)
    return source, dest, backup


def check_wan2_2_repo() -> bool:
    """Check if Wan2.2 repo exists and is valid."""
    if not WAN2_2_REPO.exists():
        print(f"Error: Wan2.2 repository not found at: {WAN2_2_REPO}")
        print("Set the WAN2_2_REPO environment variable to the correct path.")
        return False

    # Check for essential files
    if not (WAN2_2_REPO / "generate.py").exists():
        print(f"Error: {WAN2_2_REPO} does not appear to be a valid Wan2.2 repo.")
        print("Expected to find generate.py in the root directory.")
        return False

    return True


def check_patches_exist() -> bool:
    """Check if all patch files exist."""
    missing = []
    for filename in PATCH_FILES:
        source, _, _ = get_file_paths(filename)
        if not source.exists():
            missing.append(str(source))

    if missing:
        print("Error: Missing patch files:")
        for f in missing:
            print(f"  - {f}")
        return False

    return True


def patch_files() -> bool:
    """Apply patches to the Wan2.2 repository."""
    print(f"Patching Wan2.2 repository at: {WAN2_2_REPO}")
    print()

    if not check_wan2_2_repo() or not check_patches_exist():
        return False

    success = True
    for filename in PATCH_FILES:
        source, dest, backup = get_file_paths(filename)

        print(f"Patching: {filename}")

        # Create backup if original exists and no backup exists yet
        if dest.exists() and not backup.exists():
            print(f"  Creating backup: {backup.name}")
            shutil.copy2(dest, backup)
        elif dest.exists() and backup.exists():
            print(f"  Backup already exists: {backup.name}")
        elif not dest.exists():
            print(f"  Warning: Original file not found (new file will be created)")

        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy patch to destination
        try:
            shutil.copy2(source, dest)
            print(f"  Applied patch to: {dest}")
        except Exception as e:
            print(f"  Error copying file: {e}")
            success = False

    print()
    if success:
        print("Patch complete!")
    else:
        print("Patch completed with errors. Please check the messages above.")

    return success


def restore_files() -> bool:
    """Restore original files from backups."""
    print(f"Restoring original files in: {WAN2_2_REPO}")
    print()

    if not check_wan2_2_repo():
        return False

    success = True
    any_restored = False

    for filename in PATCH_FILES:
        _, dest, backup = get_file_paths(filename)

        if backup.exists():
            print(f"Restoring: {filename}")
            try:
                shutil.copy2(backup, dest)
                backup.unlink()
                print(f"  Restored from: {backup.name}")
                any_restored = True
            except Exception as e:
                print(f"  Error restoring file: {e}")
                success = False
        else:
            print(f"Skipping: {filename} (no backup found)")

    print()
    if not any_restored:
        print("No backups found. Nothing to restore.")
        print("The Wan2.2 repo may not have been patched, or backups were removed.")
    elif success:
        print("Restore complete!")
    else:
        print("Restore completed with errors. Please check the messages above.")

    return success


def check_status() -> None:
    """Check the current patch status."""
    print(f"Wan2.2 repository: {WAN2_2_REPO}")
    print(f"Patches directory: {PATCHES_DIR}")
    print()

    if not WAN2_2_REPO.exists():
        print("Status: Wan2.2 repository NOT FOUND")
        return

    print("File status:")
    print("-" * 60)

    for filename in PATCH_FILES:
        source, dest, backup = get_file_paths(filename)

        source_exists = source.exists()
        dest_exists = dest.exists()
        backup_exists = backup.exists()

        # Determine status
        if not source_exists:
            status = "MISSING PATCH"
        elif not dest_exists:
            status = "NOT APPLIED (target missing)"
        elif backup_exists:
            # Compare source with dest to see if patch is current
            if source.read_bytes() == dest.read_bytes():
                status = "PATCHED"
            else:
                status = "OUTDATED (patch differs from target)"
        else:
            # No backup - check if files match
            if source.read_bytes() == dest.read_bytes():
                status = "PATCHED (no backup)"
            else:
                status = "UNPATCHED"

        print(f"  {filename}: {status}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Patch Wan2.2 with frontend modifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  patch     Apply patches to the Wan2.2 repository
  restore   Restore original files from backups
  status    Check current patch status

Environment Variables:
  WAN2_2_REPO   Path to the Wan2.2 repository (default: ../Wan2.2)
        """
    )

    parser.add_argument(
        "command",
        choices=["patch", "restore", "status"],
        help="Command to execute"
    )

    parser.add_argument(
        "--wan-repo",
        type=Path,
        help="Override the Wan2.2 repository path"
    )

    args = parser.parse_args()

    # Override repo path if specified
    global WAN2_2_REPO
    if args.wan_repo:
        WAN2_2_REPO = args.wan_repo.resolve()

    # Execute command
    if args.command == "patch":
        success = patch_files()
        sys.exit(0 if success else 1)
    elif args.command == "restore":
        success = restore_files()
        sys.exit(0 if success else 1)
    elif args.command == "status":
        check_status()
        sys.exit(0)


if __name__ == "__main__":
    main()
