#!/bin/bash

# Function to increment version
increment_version() {
    local version=$(cat VERSION)
    IFS='.' read -ra VERSION_PARTS <<< "$version"
    local major=${VERSION_PARTS[0]}
    local minor=${VERSION_PARTS[1]}
    local patch=${VERSION_PARTS[2]}

    case $1 in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            echo "Invalid part specified. Must be one of 'major', 'minor', or 'patch'."
            exit 1
            ;;
    esac

    echo "$major.$minor.$patch"
}

# Parse command-line arguments
part=${1:-patch}

echo "Starting the version bump script; updating part $part"

# Increment the version
new_version=$(increment_version "$part")
echo "$new_version" > VERSION
echo "Version incremented to $new_version"