from pathlib import Path

# Check what's in CoRR_data
corr_path = Path('CoRR_data')

print("=== Contents of CoRR_data ===")
for item in sorted(corr_path.iterdir())[:10]:  # Show first 10 items
    print(item)

print("\n=== Looking for subject folders ===")
sub_folders = list(corr_path.glob('sub-*'))
print(f"Found {len(sub_folders)} subject folders")

if sub_folders:
    # Check first subject
    first_sub = sub_folders[0]
    print(f"\n=== Contents of {first_sub.name} ===")
    for item in first_sub.rglob('*'):
        if item.is_file():
            print(item.relative_to(corr_path))