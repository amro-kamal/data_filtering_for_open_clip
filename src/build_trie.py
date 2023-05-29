import pygtrie
from tqdm import tqdm
import pickle


def stringtrie(arr):
    trie = pygtrie.StringTrie()
    i = 0
    for p in tqdm(arr):
        key = p.strip()[:-4]
        trie[key] = 1
        i += 1
    print(f"saved {i} strings")
    return trie


def trie_search(trie, seq="00000/00000"):
    try:
        if trie[seq]:
            return True
    except KeyError:
        return False


def build_and_save_trie(data: list, file_path: str):
    """
      Webdataset is stored in shards. Each shard is simply a set of tar files. Each tar file usually stores a few thousand examples (images, text files, JSON files, etc). For example, LAION-440M is stored in 128 shards "shard0", "shard1", ... "shard127".

      Each shard stores multiple tar files. For example, "shard0/00000.tar", ...., "shard0/00362.tar". Each tar files stores images, captions, json files and other meta data files. Each tar file stores images, captions, JSON files, and other metadata files. To load data from these files, you need to untar them first.

     For simplicity, let's write the paths to data examples in this way: "shard0/00000.tar/key.jpg", "shard0/00000.tar/key.txt", "shard0/00000.tar/key.json". The unique identifier of each example could be this part from its path: "shard0/00000.tar/key.jpg" OR simply "0/00000/key.jpg" as "<shard_id/tar/key>".

    To filter LAION, we store the example IDs we need to load in the format "<shard_id/tar/key>" in a file. Later on, we will use an example if its ID exists in the file (we search for the example in the file).

    For small datasets like ImageNet, we can store these IDs in a text file. However, for LAION (millions of IDs), we need a fast search since searching in text files or lists is very slow. For that reason, we store the IDs (strings of the format "<shard_id/tar/key>") in a trie data structure, which is very fast for search.

    Note: You need to know which examples you need to filter/keep in order to build the trie data structure. In other words, you need to have IDs ("<shard_id/tar/key>") stored in a list or text file.

    Note: Another way to filter examples is to loop through all examples in the text file and load them all without searching for example IDs in the file. Unfortunately, this solution has many challenges and is very slow. For example:

    1. We will need to untar a tar file to load a single example only.
    2. We can solve (1) by sorting examples in the text file by their shards, but shuffling will be hard to implement in this case. Can overriding webdataset shuffling function help here? Not sure.


    """
    print("Building trie structure for the pruned data")
    mytrie = stringtrie(data)
    print("Saving trie.pickle ...")
    with open(file_path, "wb") as f:
        pickle.dump(mytrie, f)
    print("Saved")

    return
