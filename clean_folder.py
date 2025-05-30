import os
import sys
import shutil

def copy_folder_contents(source_dir: str, dest_dir: str, prefix: str):
    """
    Copy all files from source_dir into dest_dir with a specific prefix.
    Creates dest_dir if it doesn't exist.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist or is not a directory: {source_dir}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Copying files from\n  {source_dir}\n→ {dest_dir}\n")
    counter = 0
    for fname in os.listdir(source_dir):
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(dest_dir, f"{prefix}{counter}.json")
        counter += 1

        # Only copy files (skip directories)
        if os.path.isfile(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {fname} → {prefix}{counter-1}.json")
            except Exception as e:
                print(f"Failed to copy {fname}: {e}")
        else:
            print(f"Skipping (not a file): {fname}")

    print(f"Done with {source_dir}. Copied {counter} files.\n")
    return counter

def process_parent_folder(parent_folder: str, dest_dir: str):
    """
    Process all subfolders in the parent folder, copying their contents
    to the destination directory with appropriate prefixes.
    """
    if not os.path.isdir(parent_folder):
        print(f"Error: parent folder does not exist: {parent_folder}")
        sys.exit(1)
        
    os.makedirs(dest_dir, exist_ok=True)
    
    total_files = 0
    subfolder_count = 0
    
    # List all subfolders
    subfolders = [f for f in os.listdir(parent_folder) 
                 if os.path.isdir(os.path.join(parent_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in {parent_folder}")
        return
        
    print(f"Found {len(subfolders)} subfolders to process")
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_count += 1
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # Create a unique prefix for this subfolder (first letter + number)
        prefix = f"{subfolder[0].upper()}{subfolder_count}_"
        
        # Copy the contents
        files_copied = copy_folder_contents(subfolder_path, dest_dir, prefix)
        total_files += files_copied
    
    print(f"\nCompleted processing all subfolders.")
    print(f"Total subfolders processed: {subfolder_count}")
    print(f"Total files copied: {total_files}")

def main():
    parent_folder = 'cleanedjsons'  # Change this to your parent folder
    dest_dir = 'attempts_jsons'      # Change this to your destination
    
    process_parent_folder(parent_folder, dest_dir)

if __name__ == "__main__":
    main()