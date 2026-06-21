#!/bin/bash

# ==============================================================================
# 🐧 Linux Command Toolkit for Machine Learning
#
# This script is a practical tool you can use on any Linux server.
# It automatically sets up a standard ML project directory structure,
# checks system resources (CPU, RAM, GPU), and creates dummy data for testing.
# ==============================================================================

# Stop execution if any command fails
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   ML Project & System Assessment Toolkit     ${NC}"
echo -e "${BLUE}==============================================${NC}\n"

# ---------------------------------------------------------
# 1. System Assessment
# ---------------------------------------------------------
echo -e "${YELLOW}>> Running System Resource Assessment...${NC}"

# Get CPU info
CPU_CORES=$(nproc)
echo -e "   ${GREEN}CPU Cores:${NC} $CPU_CORES"

# Get RAM info (awk is used to parse the 'free -h' output)
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | grep Mem | awk '{print $2}')
    FREE_RAM=$(free -h | grep Mem | awk '{print $4}')
    echo -e "   ${GREEN}RAM:${NC} Total: $TOTAL_RAM | Free: $FREE_RAM"
else
    echo -e "   ${RED}RAM check skipped (free command not found, usually on Mac)${NC}"
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "   ${GREEN}GPU Detected:${NC} $GPU_NAME"
else
    echo -e "   ${RED}No NVIDIA GPU detected or nvidia-smi not installed.${NC}"
fi

echo ""

# ---------------------------------------------------------
# 2. Automated Project Scaffolding
# ---------------------------------------------------------
PROJECT_NAME="ml_project_workspace"

echo -e "${YELLOW}>> Scaffolding ML Project Directory: ${PROJECT_NAME}...${NC}"

if [ -d "$PROJECT_NAME" ]; then
    echo -e "   ${RED}Directory '$PROJECT_NAME' already exists. Cleaning it up...${NC}"
    rm -rf "$PROJECT_NAME"
fi

# Create directory tree
mkdir -p "$PROJECT_NAME"/{data/raw,data/processed,notebooks,src/models,src/data,models,logs}

# Create standard empty files
touch "$PROJECT_NAME"/README.md
touch "$PROJECT_NAME"/requirements.txt
touch "$PROJECT_NAME"/.gitignore
touch "$PROJECT_NAME"/src/__init__.py
touch "$PROJECT_NAME"/src/data/__init__.py
touch "$PROJECT_NAME"/src/models/__init__.py

echo -e "   ${GREEN}✓ Project structure created successfully.${NC}"

# ---------------------------------------------------------
# 3. Generating Mock Data using Shell Commands
# ---------------------------------------------------------
echo -e "\n${YELLOW}>> Generating mock CSV dataset...${NC}"

DATA_FILE="$PROJECT_NAME/data/raw/mock_dataset.csv"

# Add header
echo "id,feature_1,feature_2,label" > "$DATA_FILE"

# Loop to generate 100 rows of random data
for i in {1..100}
do
    # Generate random numbers using $RANDOM (0-32767)
    F1=$((RANDOM % 100))
    F2=$((RANDOM % 100))
    
    # Simple rule for label: 1 if F1 + F2 > 100, else 0
    if [ $((F1 + F2)) -gt 100 ]; then
        LABEL=1
    else
        LABEL=0
    fi
    
    echo "$i,$F1,$F2,$LABEL" >> "$DATA_FILE"
done

echo -e "   ${GREEN}✓ Generated 100 rows of mock data at: $DATA_FILE${NC}"

# Use 'head' to show the first 5 lines
echo -e "\n${BLUE}   Preview of generated data:${NC}"
head -n 5 "$DATA_FILE"

# ---------------------------------------------------------
# 4. Final Instructions
# ---------------------------------------------------------
echo -e "\n${BLUE}==============================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "To start working, navigate to your new directory:"
echo -e "   ${YELLOW}cd $PROJECT_NAME${NC}"
echo -e "   ${YELLOW}ls -la${NC}"
echo -e "${BLUE}==============================================${NC}"
