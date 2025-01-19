#!/bin/bash

# Generate tree excluding common unnecessary files
tree -I 'node_modules|.git|.DS_Store|__pycache__|*.pyc' > repo_structure.txt

# Extract tree content and update README.md
awk '/## Repository Structure/{p=NR+1}p==NR{next} /```/{exit} {print}' README.md > temp.md
echo "## Repository Structure" >> temp.md
echo "\`\`\`bash" >> temp.md
cat repo_structure.txt >> temp.md
echo "\`\`\`" >> temp.md
awk '!found && /## Repository Structure/{found=1;next} found&&/```/{found=0;next} !found' README.md >> temp.md
mv temp.md README.md

# Clean up
rm repo_structure.txt

