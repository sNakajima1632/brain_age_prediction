# Define source and destination directories
$src = "C:\Users\shido\Documents\_utd\utd_grad\0_2025Fall\BMEN6367_AIinBmen\project\IXIprep"
$dst = "C:\Users\shido\Documents\_utd\utd_grad\0_2025Fall\BMEN6367_AIinBmen\project\IXIprep_final_image_only"

# Create the destination folder if it doesn't exist
New-Item -ItemType Directory -Path $dst -Force | Out-Null

# Find all files that match *SRI*.nii.gz recursively under $src
Get-ChildItem -Path $src -Recurse -Filter "*SRI*.nii.gz" | ForEach-Object {
    # $_.FullName  → full path of the file
    # $_.DirectoryName → folder containing the file

    # Derive relative path (e.g., "1\..." from folder1\1\...)
    $relative = $_.FullName.Substring($src.Length).TrimStart('\')
    $subject = $relative.Split('\')[0]

    # Create a subject folder in destination
    $targetDir = Join-Path $dst $subject
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }

    # Copy the file into the corresponding subject folder
    Copy-Item -LiteralPath $_.FullName -Destination $targetDir -Force
}

Write-Host "✅ Done copying all *SRI*.nii.gz files into subfolders under $dst"
