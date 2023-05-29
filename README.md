# OpenCLIP
The Code here is a clone from [OpenCLIP repo (Ilharco et.el 2022)](https://github.com/mlfoundations/open_clip). The src/training/data.py file was modified to support filtering with [Webdataset](https://github.com/webdataset/webdataset) loaders.


## Webdataset

  Webdataset is stored in shards. Each shard is simply a set of tar files. Each tar file usually stores a few thousand examples (images, text files, JSON files, etc). For example, LAION-440M is stored in 128 shards "shard0", "shard1", ... "shard127".

  Each shard stores multiple tar files. For example, "shard0/00000.tar", ...., "shard0/00362.tar". Each tar files stores images, captions, json files and other meta data files. Each tar file stores images, captions, JSON files, and other metadata files. To load data from these files, you need to untar them first.

 For simplicity, let's write the paths to data examples in this way: "shard0/00000.tar/key.jpg", "shard0/00000.tar/key.txt", "shard0/00000.tar/key.json". The unique identifier of each example could be this part from its path: "shard0/00000.tar/key.jpg" OR simply "0/00000/key.jpg" as `"<shard_id/tar/key>"`.

To filter LAION, we store the example IDs we need to load in the format `"<shard_id/tar/key>"` in a file. Later on, we will use an example if its ID exists in the file (we search for the example in the file).

For small datasets like ImageNet, we can store these IDs in a text file. However, for LAION (millions of IDs), we need a fast search since searching in text files or lists is very slow. For that reason, we store the IDs (strings of the format `"<shard_id/tar/key>"`) in a trie data structure, which is very fast for search.

Note: the format `"<shard_id/tar/key>"` depends on how data is structured and stored. Some people put all tar files in a single shard. In this case you can drop the shard_id part.
Note: You need to know which examples you need to filter/keep in order to build the trie data structure. In other words, you need to have IDs (`"<shard_id/tar/key>"`) stored in a list or text file.

Note: Another way to filter examples is to loop through all examples in the text file and load them all without searching for example IDs in the file. Unfortunately, this solution has many challenges and is very slow. For example:

1. We will need to untar a tar file to load a single example only.
2. We can solve (1) by sorting examples in the text file by their shards, but shuffling will be hard to implement in this case. Can overriding webdataset shuffling function help here? Not sure.


## Build trie file
Assuming you have a list of examples in the format [`"<shard_id/tar/key>"`, `"<shard_id/tar/key>"`, `"<shard_id/tar/key>"`, `"<shard_id/tar/key>"`]. Run the following code to convert the list into a trie structure and store it as a pickle file.

```python
from build_trie import build_and_save_trie
list_of_ids = [....]
trie_file = ".....pickle"
build_and_save_trie(list_of_ids, trie_file)

```


The current version of the code assume that each example has the format `"<shard_id/tar/key>"`. This is hard-coded in the `get_file_id(.)` in `src/training/data.py`

```python

def get_file_id(url, fname):

    '''
    url: path to tar file where the example is stored.  This function expects that each url is in this format "LAION/laion2B-en-joined{0..127}/<tar file>". This fucntion consider the unique part of the shard name only, which is the value of {0..127} (i.e shard[17:]). We do that because it is important to make the string short.
    fname: name of the example in the tar file. Each pair of image, text (caption) share the same fname.

    '''
    file_key = (
        url.split(".")[0].split("/")[-2][17:] ## -- shard name: LAION/laion2B-en-joined{0..127}/
        + "/"
        + url.split(".")[0].split("/")[-1]    ## -- tar file
        + "/"
        + fname.split(".")[0] ## -- example key
    )
    return file_key

file_key = get_file_id(url, fname)
```
### IMPOTANT:
For now, you need to modify the `get_file_id(.)` in `src/training/data.py` to work with your dataset. You simply need to know how the dataset is structured.

TODO-SOON: modify the code, to enable the use to pass this function as a argument instead modifying it inside `src/training/data.py`



## How to train OpenCLIP on filtered data
Same as OpenCLIP training script. Use the script below:

```
EPOCHS=32

EXP_NAME="openclip-b16-50%-rand"

FULL_DATASET_SIZE=400000000

PRUNING_RATIO=0.5 # size of the filtered data relative to the size of the whole data

TRIE_FILE=...  # trie file stored as a pickle file (output of build_and_save_trie(.))

BATCH_SIZE=...

srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --subset-file ${TRIE_FILE} \
    --prune-ratio ${PRUNING_RATIO} \
    --save-frequency 1 \
    --report-to wandb \
    --train-data "/shards/laion2B-en-joined{0..127}/{00000..00362}.tar" \
    --warmup 2000 \
    --batch-size ${BATCH_SIZE} \
    --dataset-type webdataset \
    --epochs ${EPOCHS} \
    --train-num-samples ${FULL_DATASET_SIZE} \
    --workers 6 \
    --model ViT-B-16 \
    --seed 0 \
    --name ${EXP_NAME} \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --log-local \
    --logs "/checkpoints/${EXP_NAME}" \
    --resume None

```


### IMPORTANT:
To traing without any filtering (whole data), set `PRUNING_RATIO=1.0` and `TRIE_FILE=""`.

FULL_DATASET_SIZE is always the size of the total number of examples on the dataset. The code will automatically calculate the size of the dataset after filtering as `PRUNING_RATIO*FULL_DATASET_SIZE`. For example, for LAION-400M, FULL_DATASET_SIZE might be 400000000.
