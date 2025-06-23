#!/bin/bash

# Set color output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get absolute path of current directory
ASSETS_PATH=$(pwd)
echo -e "${BLUE}Current path:${NC} $ASSETS_PATH"

# Check if assets directory exists
if [ ! -d "./assets/embodiments" ]; then
  echo -e "${YELLOW}Warning:${NC} ./assets/embodiments directory not found"
  
  # Check if in parent directory
  if [ -d "../assets/embodiments" ]; then
    echo "Found assets/embodiments in parent directory, switching..."
    cd ..
    ASSETS_PATH=$(pwd)
    echo -e "${BLUE}Updated path:${NC} $ASSETS_PATH"
  else
    echo -e "${YELLOW}Please ensure you're running this script in the correct directory${NC}"
    echo "Script should be run in the repository root directory containing assets/embodiments"
    
    # Ask user if they want to specify a path
    read -p "Do you want to manually specify the absolute path to the assets directory? (y/n): " answer
    if [[ "$answer" == "y" ]]; then
      read -p "Please enter the absolute path: " ASSETS_PATH
      if [ ! -d "$ASSETS_PATH/assets/embodiments" ]; then
        echo -e "${YELLOW}Error:${NC} Cannot find assets/embodiments directory at the specified path"
        exit 1
      fi
      cd $ASSETS_PATH
      echo -e "${BLUE}Switched to:${NC} $ASSETS_PATH"
    else
      exit 1
    fi
  fi
fi

# Export environment variable
export ASSETS_PATH
echo -e "${BLUE}Setting environment variable:${NC} ASSETS_PATH=$ASSETS_PATH"

# Counters
count_total=0
count_updated=0
count_error=0

# Get all configuration template files
echo -e "${BLUE}Searching for configuration template files...${NC}"
CONFIG_FILES=$(find ./assets/embodiments -name "*_tmp.yml")

if [ -z "$CONFIG_FILES" ]; then
  echo -e "${YELLOW}No *_tmp.yml files found${NC}"
  exit 1
fi

# Process each configuration file
echo -e "${BLUE}Starting to process configuration files...${NC}"
for tmp_file in $CONFIG_FILES; do
  count_total=$((count_total + 1))
  
  # Get target filename (remove _tmp suffix)
  target_file="${tmp_file%_tmp.yml}.yml"
  dir_name=$(dirname "$tmp_file")
  base_name=$(basename "$target_file")
  
  echo -e "Processing [${count_total}]: ${YELLOW}$tmp_file${NC} -> ${GREEN}$target_file${NC}"
  
  # Use envsubst to replace environment variables
  if envsubst < "$tmp_file" > "$target_file"; then
    echo -e "  ${GREEN}✓${NC} Successfully replaced \${ASSETS_PATH} -> $ASSETS_PATH"
    count_updated=$((count_updated + 1))
    
    # Check if placeholders were actually replaced
    if grep -q "\${ASSETS_PATH}" "$tmp_file"; then
      if grep -q "$ASSETS_PATH" "$target_file"; then
        echo -e "  ${GREEN}✓${NC} Confirmed path was correctly replaced"
      else
        echo -e "  ${YELLOW}!${NC} Warning: Could not confirm if path was correctly replaced"
      fi
    fi
  else
    echo -e "  ${YELLOW}✗${NC} Replacement failed"
    count_error=$((count_error + 1))
  fi
done

# Results summary
echo -e "\n${BLUE}Processing complete!${NC}"
echo -e "Total processed: ${count_total} files"
echo -e "Successfully updated: ${GREEN}${count_updated}${NC} files"
if [ $count_error -gt 0 ]; then
  echo -e "Failed to process: ${YELLOW}${count_error}${NC} files"
fi

# Final instructions
echo -e "\n${GREEN}All template files have been processed!${NC}"
echo "To use in a new environment, run this script again"