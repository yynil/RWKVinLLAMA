import glob
import json
import os
from tqdm import tqdm

def convert_2_standard(input_dir, output_dir):
    """
    Convert the jsonl contents to standard jsonl contents with progress bar
    """
    # 1. Iterate the jsonl files in the input directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} jsonl files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add progress bar for files
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        # 2. Read the jsonl file and count lines
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 3. Create the same name jsonl file in the output directory
        output_file = os.path.join(output_dir, os.path.basename(jsonl_file))
        
        # Add progress bar for lines in current file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in tqdm(lines, desc=f"Processing {os.path.basename(jsonl_file)}", leave=False):
                data = json.loads(line)
                
                # 4. Process the text field if it exists
                if 'text' in data:
                    text = data['text']
                    
                    # Split by markers
                    parts = text.split('\n\n')
                    messages = []
                    
                    current_user_content = []
                    
                    for part in parts:
                        if part.startswith('System: '):
                            current_user_content.append(part[8:])  # Remove "System: "
                        elif part.startswith('User: '):
                            current_user_content.append(part[6:])  # Remove "User: "
                        elif part.startswith('Assistant: '):
                            # First, combine all previous content as user message
                            if current_user_content:
                                messages.append({
                                    "role": "user",
                                    "content": "\n\n".join(current_user_content)
                                })
                                current_user_content = []
                            
                            # Add assistant message
                            messages.append({
                                "role": "assistant",
                                "content": part[11:]  # Remove "Assistant: "
                            })
                    
                    # Handle any remaining user content
                    if current_user_content:
                        messages.append({
                            "role": "user",
                            "content": "\n\n".join(current_user_content)
                        })
                    
                    # Create the new format
                    new_data = {"messages": messages}
                    
                    # Write to output file
                    f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                else:
                    # If no text field, write the original line
                    f.write(line)
                    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="The input directory containing jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory to save the converted jsonl files")
    args = parser.parse_args()
    convert_2_standard(args.input_dir, args.output_dir)