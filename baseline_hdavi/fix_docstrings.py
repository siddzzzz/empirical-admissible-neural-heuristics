import os
import glob

for filepath in glob.glob('**/*.py', recursive=True):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if '\\"\\"\\"' in content:
        new_content = content.replace('\\"\\"\\"', '"""')
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")
