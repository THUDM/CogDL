import os.path as osp
import os
import json
from cogdl.data import Dataset
from cogdl.utils import download_url, untar


url_map = {
    "l0fos": "https://cloud.tsinghua.edu.cn/f/cd6e3f3276c14e73a9f7/?dl=1",
    "aff30": "https://cloud.tsinghua.edu.cn/f/c37df92984214b9bad66/?dl=1",
    "arxivvenue": "https://cloud.tsinghua.edu.cn/f/5bd4597726514a119898/?dl=1",
}


class oagbert_dataset(Dataset):
    def __init__(self):
        self.url = url_map[self.name]
        path = osp.join("data", self.name)
        self.content = ["._SUCCESS"]
        self.candidates = []
        super(oagbert_dataset, self).__init__(path, self.name)

    def download(self):
        filename = self.name + ".zip"
        download_url(self.url, self.processed_dir, name=filename)
        untar(self.processed_dir, filename)
        print(f"downloaded to {self.processed_dir}")

    def process(self):
        pass

    def get_candidates(self):
        """
        return the list of candidates
        """
        with open("%s/._SUCCESS" % self.processed_dir) as f:
            for line in f:
                line = line.strip()
                self.candidates.append(line)
        return self.candidates

    def get_data(self):
        """
        Return all the file name
        Not expanded to the other two dataset yet
        """
        sample = {}
        for filename in os.listdir(self.processed_dir):
            if not filename.endswith(".jsonl"):
                continue
            sample_file = self.processed_dir + "/" + filename
            fin = open(sample_file, "r")
            file = []
            for line in fin:
                paper = json.loads(line.strip())
                file.append(paper)
            sample[filename] = file
        return sample

    @property
    def raw_file_names(self):
        return self.content

    def __len__(self):
        return 20

    @property
    def processed_file_names(self):
        return self.content


class l0fos(oagbert_dataset):
    def __init__(self):
        self.name = "l0fos"
        super(l0fos, self).__init__()


class aff30(oagbert_dataset):
    def __init__(self):
        self.name = "aff30"
        super(aff30, self).__init__()


class arxivvenue(oagbert_dataset):
    def __init__(self):
        self.name = "arxivvenue"
        super(arxivvenue, self).__init__()
