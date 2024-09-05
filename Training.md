# Corpus preparation
We use different corpus for different stage of training.
For stage 1, we use allenai/c4 dataset's english and chinese corpus.
For stage 2, we use a higher quality corpus which contains more diverse data including smollm/cospedia-v2, chinese-fineweb, baai/IndustryCorpus,fineweb-edu-mini-combined, smollm/pyedu.

1. Download corpus
Please download the corpus from huggingface or other places.
2. Preprocess corpus
Some of the corpus is jsonl or jsonl.gz file like baai/IndustryCorpus. Use the following command to convert it to parquet file.
Since IndustryCorpus has a lot of kinds of data, we can only use part of it. Here is an example to only use the industry data.
```bash
DATA_DIR="data/IndustryCorpus"
OUTPUT_DIR="data/IndustryCorpus/parquet"
python data/multi_source_datasets.py --paths ${DATA_DIR}/IndustryCorpus/zh/program/ ${DATA_DIR}/IndustryCorpus/zh/edu/ ${DATA_DIR}/IndustryCorpus/zh/politic/ --convert_to_parquet --output_dirs ${OUTPUT_DIR}/zh/program ${OUTPUT_DIR}/zh/edu ${OUTPUT_DIR}/zh/politic --file_extension .jsonl.gz
```
This commands will generate the parquet files in the output directory which contains program, edu and politic data. THE ONLY REQUIREMENT of the parquet file is that it should have a column named "text".  

3. Combine all the parquet files
Suppose we have several directories like OUTPUT_DIR/zh/program OUTPUT_DIR/zh/edu OUTPUT_DIR/zh/politic which contains parquet files for different data. We need to combine,tokenize,padding and truncate them to a fixed length. After preprocessing data like above, we will get a high efficient dataset for training. 
```bash
python data/multi_source_datasets.py --paths /data/rwkv/data/fineweb-edu-mini-combined/train/ /data/rwkv/data/IndustryCorpus/IndustryCorpus/parquet/zh/ /data/rwkv/data/pyedu/ /data/rwkv/data/smollm-corpus/cosmopedia-v2/ /data/rwkv/data/chinese-fineweb-edu/data --is_save_to_disk --output_dirs /data/rwkv/data/stage2/train /data/rwkv/data/stage2/val --model_id /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --max_seq_length 2048
```
After running above command, we will get a dataset with a fixed length(2048) with input_ids and labels which are right shifted by one from the input_ids. This command will also generate a dataset for validation.

4. Split the dataset by different lengths. We can split the dataset by different lengths to utilize the data more efficiently.
```bash
DATA_DIR="/data/rwkv/data/stage2/"
OUTPUT_DIR="/data/rwkv/data/stage2/sub_datasets"
python data/select_sub_dataset.py --input_dir ${DATA_DIR}/train/ --model_name /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --min_len 1 --max_len 2048 --output_dir ${OUTPUT_DIR}/train/ --step 256
python data/select_sub_dataset.py --input_dir ${DATA_DIR}/val/ --model_name /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --min_len 1 --max_len 2048 --output_dir ${OUTPUT_DIR}_val/ --step 256
```
This command will generate several sub-datasets with different lengths with step 256 from the original dataset. The sub-datasets's length is [1,256],[257,512],[513,768],[769,1024],[1025,1280],[1281,1536],[1537,1792],[1793,2048].

We will get the final dataset like below:

train:
```
$OUTPUT_DIR_1_256
|
$OUTPUT_DIR_257_512
|
$OUTPUT_DIR_513_768
|
$OUTPUT_DIR_769_1024
|
$OUTPUT_DIR_1025_1280
|
$OUTPUT_DIR_1281_1536
|
$OUTPUT_DIR_513_768
|
$OUTPUT_DIR_769_1024
|
$OUTPUT_DIR_1025_1280
|
$OUTPUT_DIR_1281_1536
|
$OUTPUT_DIR_1537_1792
|
$OUTPUT_DIR_1793_2048
```
val:
```
$OUTPUT_DIR_val_1_256
|
$OUTPUT_DIR_val_257_512
|
$OUTPUT_DIR_val_513_768
|
$OUTPUT_DIR_val_769_1024
|
$OUTPUT_DIR_val_1025_1280
|
$OUTPUT_DIR_val_1281_1536
|
$OUTPUT_DIR_val_1537_1792
|
$OUTPUT_DIR_val_1793_2048
```
# Stage 1 Training

# Stage 2 Training
We can run stage 2 training with different sub-datasets. For example, we can run stage 2 training with sub-dataset 1_256 and sub-dataset 257_512 with following command:
```bash
sh train_scripts/train.sh -m 1 -M 256 -b 48 -w 100 -v 5000 -n 6 
```
This command will run stage 2 training with sub-dataset 1_256 and sub-dataset 257_512.