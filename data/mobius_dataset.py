import argparse
from transformers import AutoTokenizer
import glob
import json
max_seq_length = 4096
def convert_str_to_list(data):
    '''
    Convert string to list
    Sample data is :
    "User: What are two potential healthcare applications of high-definition video frame extraction for real-time processing?\n\nAssistant: Two potential healthcare applications of high-definition video frame extraction for real-time processing include: 1. Telemedicine and remote patient monitoring, where high-definition video can be used to assess patient conditions and provide real-time feedback to healthcare providers. 2. Surgical assistance and training, where extracting frames from high-definition video feeds can help in detailed analysis of surgical procedures for educational purposes or to assist surgeons in making real-time decisions during operations.\n\nUser: What are the applications of extracting frames from videos in retail analytics and wildlife monitoring that require interactive video content and user feedback incorporation, while also complying with GDPR for data privacy?\n\nAssistant: In retail analytics, extracting frames from videos can be used for applications such as analyzing customer behavior, monitoring inventory levels, and optimizing store layouts. Interactive video content can engage customers by providing personalized shopping experiences, while user feedback can be incorporated to improve service quality and customer satisfaction. In wildlife monitoring, frame extraction can help in species identification, behavior analysis, and population tracking. Interactive content can be used for educational purposes or to engage the public in conservation efforts, and user feedback can inform better wildlife management practices. Both applications must ensure that any personal data captured is processed in compliance with GDPR, which may involve anonymizing individuals in retail settings and ensuring that any data collected in wildlife monitoring does not inadvertently capture or store personal data without consent.\n\nUser: How does high-definition video frame extraction with real-time processing and low latency contribute to motion analysis and object tracking in the domains of healthcare, sports, and security, ensuring data encryption and accessibility features? List in ordered list, max 5 applications per domain.\n\nAssistant: High-definition video frame extraction with real-time processing and low latency contributes to motion analysis and object tracking in the domains of healthcare, sports, and security in the following ways, ensuring data encryption and accessibility features:\nHealthcare:\n1. Enables precise monitoring of patient movements for physical therapy and rehabilitation, aiding in the assessment of recovery progress.\n2. Facilitates the detection of abnormal movements or falls in elderly care environments, triggering immediate alerts for assistance.\n3. Allows for real-time surgical training and assistance by providing detailed visualizations of complex procedures.\n4. Supports remote patient monitoring systems, ensuring patient data is securely transmitted and accessible to authorized medical personnel.\n5. Enhances diagnostic procedures by providing high-resolution images for analysis of patient's gait and posture.\n\nSports:\n1. Improves athlete performance analysis by capturing detailed movements for coaching and training feedback.\n2. Enables real-time strategy adjustments by providing coaches with immediate visual data during games.\n3. Assists in injury prevention by analyzing athletes' movements to identify potential risk patterns.\n4. Enhances fan experience by providing high-definition replays and analyses during live broadcasts.\n5. Secures sensitive team data during video analysis sessions through encryption, ensuring only authorized personnel have access.\n\nSecurity:\n1. Enhances surveillance systems by allowing for the clear identification of individuals and objects in real-time.\n2. Improves perimeter defense by quickly detecting and tracking intrusions with minimal delay.\n3. Aids in crowd monitoring and management by analyzing movement patterns to prevent incidents.\n4. Supports forensic analysis by providing high-quality video evidence that can be securely stored and accessed.\n5. Enables automatic threat detection by integrating with AI algorithms that can process high-definition video in real time.\n\n"
    '''
    data = data.strip()
    system_str = 'System: '
    user_str = 'User: '
    assistant_str = 'Assistant: '
    double_new_line = '\n\n'
    conversations = []
    offset = 0
    if len(data) == 0:
        return conversations
    while True:
        #check if current user_str or assistant_str
        role_len = 0
        if data[offset:offset+len(user_str)] == user_str:
            current_role = 'user'
            next_offset = data.find(double_new_line+assistant_str, offset)
            role_len = len(user_str)
        elif data[offset:offset+len(assistant_str)] == assistant_str:
            current_role = 'assistant'
            next_offset = data.find(double_new_line+user_str, offset)
            role_len = len(assistant_str)
        elif data[offset:offset+len(system_str)] == system_str:
            current_role = 'system'
            next_offset = data.find(double_new_line+user_str, offset)
            role_len = len(system_str)
        else:
            print(f'Error: Invalid role at offset {offset} in data {data}')
        if next_offset == -1:
            content = data[offset+role_len:]
        else:
            content = data[offset+role_len:next_offset]
        conversations.append({'role':current_role, 'content':content})
        if next_offset == -1:
            break
        offset = next_offset + len(double_new_line)
    return conversations
def create_input_ids(conversations, tokenizer):
    input_ids = []
    labels = []
    input_ids.append(tokenizer.bos_token_id)
    for conv in conversations:
        role = conv['role']
        content = conv['content']
        
        if role == 'user':
            content = 'User: ' + content
        elif role == 'assistant':
            content = 'Assistant: ' + content
        elif role == 'system':
            content = 'System: ' + content
        
        if not content.endswith('\n\n'):
            content += '\n\n'
        
        tokenized = tokenizer(content, return_tensors='pt',add_special_tokens=False)
        ids = tokenized['input_ids'].squeeze().tolist()
        
        if role == 'assistant':
            input_ids.extend(ids)
            labels.extend(ids)
        else:
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))
    labels.append(tokenizer.eos_token_id)
    return input_ids, labels
def handle_jsonl(file, model_id,ranges):
    print(f'Processing {file}')
    # Create AutoTokenizer using model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(tokenizer)
    datas = [[] for i in range(len(ranges))]
    with open(file,'r',encoding='utf-8') as f:
        is_data_list = False
        index = 0
        for line in f:
            data = json.loads(line)
            if 'text' in data:
                if index == 0:
                    if isinstance(data['text'], list):
                        is_data_list = True
                if not is_data_list:
                    conversations = convert_str_to_list(data['text'])
                else:
                    conversations = data['text']
                input_ids, labels = create_input_ids(conversations, tokenizer)
                if len(input_ids) > max_seq_length:
                    input_ids = input_ids[:max_seq_length]
                    labels = labels[:max_seq_length]
                for i in range(len(ranges)):
                    if len(input_ids) <= ranges[i]:
                        datas[i].append({'input_ids':input_ids, 'labels':labels})
                        break
                index += 1
    print(f'Processed {index} samples in {file}')
    return datas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to the input directory")
    parser.add_argument("--model_id", help="Model ID for AutoTokenizer")
    parser.add_argument("--step",type=int, default=256,help="Step to process")
    parser.add_argument("--output_dir", help="Path to the output directory",type=str,required=True)
    
    args = parser.parse_args()

    input_dir = args.input_dir
    model_id = args.model_id
    ranges = []
    for i in range(args.step, max_seq_length, args.step):
        ranges.append(i)
    import os

    # Read all JSONL file names under input_dir and its subdirectories into a list
    file_names = glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True)

    print(f'All JSONL files under {input_dir} and its subdirectories: {file_names}')

    
    from multiprocessing import Pool
    with Pool(16) as p:
        datas = p.starmap(handle_jsonl, [(file, model_id,ranges) for file in file_names])
    import datasets
    from datasets import Dataset
    import os
    
    for i in range(len(ranges)):
        #merge all datas
        all_datas = []
        for data in datas:
            all_datas.extend(data[i])
        output_dir = args.output_dir+f"/length_{ranges[i]}"
        print(f'Saving to {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        ds = Dataset.from_list(all_datas)
        ds.save_to_disk(output_dir)
        print(f'Saved to {output_dir}')
if __name__ == "__main__":
    main()