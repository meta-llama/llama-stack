# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import shutil
import sys
from pathlib import Path

from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
)

from .subcommand import Subcommand


class MigrateXDG(Subcommand):
    """CLI command for migrating from legacy ~/.llama to XDG-compliant directories."""
    
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "migrate-xdg",
            prog="llama migrate-xdg",
            description="Migrate from legacy ~/.llama to XDG-compliant directories",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        
        self.parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without actually moving files"
        )
        
        self.parser.set_defaults(func=self._run_migrate_xdg_cmd)
    
    @staticmethod
    def create(subparsers: argparse._SubParsersAction):
        return MigrateXDG(subparsers)
    
    def _run_migrate_xdg_cmd(self, args: argparse.Namespace) -> None:
        """Run the migrate-xdg command."""
        if not migrate_to_xdg(dry_run=args.dry_run):
            sys.exit(1)


def migrate_to_xdg(dry_run: bool = False) -> bool:
    """
    Migrate from legacy ~/.llama to XDG-compliant directories.
    
    Args:
        dry_run: If True, only show what would be done without actually moving files
        
    Returns:
        bool: True if migration was successful or not needed, False otherwise
    """
    legacy_path = Path.home() / ".llama"
    
    if not legacy_path.exists():
        print("No legacy ~/.llama directory found. Nothing to migrate.")
        return True
    
    # Check if we're already using XDG paths
    config_dir = get_llama_stack_config_dir()
    data_dir = get_llama_stack_data_dir()
    state_dir = get_llama_stack_state_dir()
    
    if str(config_dir) == str(legacy_path):
        print("Already using legacy directory. No migration needed.")
        return True
    
    print(f"Found legacy directory at: {legacy_path}")
    print(f"Will migrate to XDG-compliant directories:")
    print(f"  Config: {config_dir}")
    print(f"  Data: {data_dir}")
    print(f"  State: {state_dir}")
    print()
    
    # Define migration mapping
    migrations = [
        # (source_subdir, target_base_dir, description)
        ("distributions", config_dir, "Distribution configurations"),
        ("providers.d", config_dir, "External provider configurations"),
        ("checkpoints", data_dir, "Model checkpoints"),
        ("runtime", state_dir, "Runtime state files"),
    ]
    
    # Check what needs to be migrated
    items_to_migrate = []
    for subdir, target_base, description in migrations:
        source_path = legacy_path / subdir
        if source_path.exists():
            target_path = target_base / subdir
            items_to_migrate.append((source_path, target_path, description))
    
    if not items_to_migrate:
        print("No items found to migrate.")
        return True
    
    print("Items to migrate:")
    for source_path, target_path, description in items_to_migrate:
        print(f"  {description}: {source_path} -> {target_path}")
    
    if dry_run:
        print("\nDry run mode: No files will be moved.")
        return True
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with the migration? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Migration cancelled.")
        return False
    
    # Perform the migration
    print("\nMigrating files...")
    
    for source_path, target_path, description in items_to_migrate:
        try:
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if target already exists
            if target_path.exists():
                print(f"  Warning: Target already exists: {target_path}")
                print(f"  Skipping {description}")
                continue
            
            # Move the directory
            shutil.move(str(source_path), str(target_path))
            print(f"  Moved {description}: {source_path} -> {target_path}")
            
        except Exception as e:
            print(f"  Error migrating {description}: {e}")
            return False
    
    # Check if legacy directory is now empty (except for hidden files)
    remaining_items = [item for item in legacy_path.iterdir() if not item.name.startswith('.')]
    if not remaining_items:
        print(f"\nMigration complete! Legacy directory {legacy_path} is now empty.")
        response = input("Remove empty legacy directory? (y/N): ")
        if response.lower() in ['y', 'yes']:
            try:
                shutil.rmtree(legacy_path)
                print(f"Removed empty legacy directory: {legacy_path}")
            except Exception as e:
                print(f"Could not remove legacy directory: {e}")
    else:
        print(f"\nMigration complete! Some items remain in legacy directory: {remaining_items}")
    
    print("\nMigration successful!")
    print("You may need to update any custom scripts or configurations that reference the old paths.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate from legacy ~/.llama to XDG-compliant directories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files"
    )
    
    args = parser.parse_args()
    
    if not migrate_to_xdg(dry_run=args.dry_run):
        sys.exit(1)


if __name__ == "__main__":
    main() 