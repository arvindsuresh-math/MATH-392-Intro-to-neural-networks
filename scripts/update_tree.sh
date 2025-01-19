#!/bin/bash

# Set output file
TREE_FILE="REPOSITORY_STRUCTURE.md"

# Generate header with timestamp
echo "# Repository Structure" > $TREE_FILE
echo "Last updated: $(date)" >> $TREE_FILE
echo "" >> $TREE_FILE

# Add tree structure in code block
echo "\`\`\`bash" >> $TREE_FILE

# List directories (all depths)
tree -d -I 'node_modules|.git|__pycache__' >> $TREE_FILE

echo "" >> $TREE_FILE
echo "Files in root directory:" >> $TREE_FILE
ls -p | grep -v / >> $TREE_FILE

echo "\`\`\`" >> $TREE_FILE

echo "Repository structure updated in $TREE_FILE"