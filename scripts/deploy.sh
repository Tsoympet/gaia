#!/bin/bash

# deploy.sh - Deployment script for G.A.I.A project
# This script sets up the environment, installs dependencies, builds the browser extension,
# and prepares the Windows installer.

# Exit on any error
set -e

# Configuration
PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
EXTENSION_DIR="$PROJECT_DIR/src/extension"
OUTPUT_DIR="$PROJECT_DIR/dist"
LOG_FILE="$PROJECT_DIR/logs/deploy.log"
INNO_SETUP_SCRIPT="$PROJECT_DIR/scripts/setup.iss"
INNO_SETUP_COMPILER="/c/Program Files (x86)/Inno Setup 6/ISCC.exe" # Adjust path if needed
GAIA_SCRIPT="$PROJECT_DIR/src/core/gaia.py"
VERSION="2.4.0"

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"
touch "$LOG_FILE"

# Logging function
log() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Error handling function
error_exit() {
    local message="$1"
    log "ERROR: $message"
    exit 1
}

# Check for required tools
check_tools() {
    log "Checking required tools..."
    command -v python3 >/dev/null 2>&1 || error_exit "Python3 is required but not installed."
    command -v pip >/dev/null 2>&1 || error_exit "pip is required but not installed."
    command -v zip >/dev/null 2>&1 || error_exit "zip is required but not installed."
    if [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then
        [[ -f "$INNO_SETUP_COMPILER" ]] || error_exit "Inno Setup compiler not found at $INNO_SETUP_COMPILER."
    fi
    log "All required tools are available."
}

# Set up virtual environment
setup_venv() {
    log "Setting up virtual environment..."
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR" || error_exit "Failed to create virtual environment."
    fi
    source "$VENV_DIR/bin/activate" || source "$VENV_DIR/Scripts/activate" || error_exit "Failed to activate virtual environment."
    log "Virtual environment activated."
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies from $REQUIREMENTS_FILE..."
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        pip install --upgrade pip || error_exit "Failed to upgrade pip."
        pip install -r "$REQUIREMENTS_FILE" || error_exit "Failed to install dependencies."
        log "Dependencies installed successfully."
    else
        error_exit "Requirements file not found at $REQUIREMENTS_FILE."
    fi
}

# Validate assets
validate_assets() {
    log "Validating assets..."
    local assets=(
        "$PROJECT_DIR/assets/gaia_logo.png"
        "$PROJECT_DIR/assets/icons/icon16.png"
        "$PROJECT_DIR/assets/icons/icon48.png"
        "$PROJECT_DIR/assets/icons/icon128.png"
    )
    for asset in "${assets[@]}"; do
        [[ -f "$asset" ]] || error_exit "Missing asset: $asset"
    done
    log "All assets validated."
}

# Build browser extension
build_extension() {
    log "Building browser extension..."
    mkdir -p "$OUTPUT_DIR/extension"
    local files=(
        "$EXTENSION_DIR/manifest.json"
        "$EXTENSION_DIR/background.js"
        "$EXTENSION_DIR/content.js"
        "$EXTENSION_DIR/popup/popup.html"
        "$EXTENSION_DIR/popup/popup.js"
    )
    for file in "${files[@]}"; do
        [[ -f "$file" ]] || error_exit "Missing extension file: $file"
        cp "$file" "$OUTPUT_DIR/extension/" || error_exit "Failed to copy $file"
    done
    cp -r "$PROJECT_DIR/assets/icons" "$OUTPUT_DIR/extension/" || error_exit "Failed to copy icons."
    cd "$OUTPUT_DIR/extension"
    zip -r "../gaia_extension_v${VERSION}.zip" . || error_exit "Failed to create extension zip."
    cd "$PROJECT_DIR"
    log "Browser extension built at $OUTPUT_DIR/gaia_extension_v${VERSION}.zip."
}

# Build Windows installer (if on Windows)
build_installer() {
    if [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then
        log "Building Windows installer..."
        if [[ -f "$INNO_SETUP_SCRIPT" ]]; then
            "$INNO_SETUP_COMPILER" "$INNO_SETUP_SCRIPT" || error_exit "Inno Setup compilation failed."
            log "Windows installer built at $OUTPUT_DIR."
        else
            error_exit "Inno Setup script not found at $INNO_SETUP_SCRIPT."
        fi
    else
        log "Skipping Windows installer build (not on Windows)."
    fi
}

# Verify main script
verify_gaia() {
    log "Verifying gaia.py..."
    if [[ -f "$GAIA_SCRIPT" ]]; then
        python3 -m py_compile "$GAIA_SCRIPT" || error_exit "gaia.py contains syntax errors."
        log "gaia.py verified."
    else
        error_exit "gaia.py not found at $GAIA_SCRIPT."
    fi
}

# Start WebSocket server (mock for testing)
start_websocket() {
    log "Starting WebSocket server (mock)..."
    # Placeholder: In production, replace with actual WebSocket server start
    echo "WebSocket server started on ws://localhost:8766 (mock)" >> "$LOG_FILE"
}

# Main deployment function
main() {
    log "Starting G.A.I.A deployment (v$VERSION)..."
    check_tools
    setup_venv
    install_dependencies
    validate_assets
    verify_gaia
    build_extension
    build_installer
    start_websocket
    log "Deployment completed successfully."
}

# Execute main
main

# Clean up
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate 2>/dev/null || true
fi
