# Query-oriented Micro-video Summarization

Query-oriented micro-video summarization task aims to generate a concise sentence with two properties: (a) summarizing the main semantic of the micro-video and (b) being expressed in the form of search queries to facilitate retrieval. 
Despite its enormous application value in the retrieval area, this direction has barely been explored. Previous studies of summarization mostly focus on the content summarization for traditional long videos. 
Directly applying these studies is prone to gain unsatisfactory results because of the unique features of micro-videos and queries: diverse entities and complex scenes within a short time, semantic gaps between modalities, and various queries in distinct expressions. 
To specifically adapt to these characteristics, we propose a query-oriented micro-video summarization model, dubbed QMS. 
It employs an encoder-decoder-based transformer architecture as the skeleton. The multi-modal (visual and textual) signals are passed through two modal-specific encoders to obtain their representations, followed by an entity-aware representation learning module to identify and highlight critical entity information. 
As to the optimization, regarding the large semantic gaps between modalities, we assign different confidence scores according to their semantic relevance in the optimization process. 
Additionally, we develop a novel strategy to sample the effective target query among the diverse query set with various expressions. Extensive experiments demonstrate the superiority of the QMS scheme, on both the summarization and retrieval tasks, over several state-of-the-art methods. 

# Getting Started
### 1. Installation
Git clone our repository and creating conda environment:

```
git clone https://github.com/Jill0001/QMS.git
cd QMS
conda create -n QMS python=3.9
conda activate QMS
pip install -r requirements.txt
```

### 2. Dataset
Visual features can be downloaded [here](https://drive.google.com/file/d/1MnBG2t2fjJzPW3bZ7uP76lPLBLKxExZq/view?usp=sharing).

Textual transcripts and queries can be downloaded [here](https://drive.google.com/file/d/1MnBG2t2fjJzPW3bZ7uP76lPLBLKxExZq/view?usp=sharing)

### 3. Checkpoints
A trained checkpoint can be downloaded [here](https://drive.google.com/file/d/1QWIMyGwYmvLZ2-u72vpHtsCr0voSnchd/view?usp=share_link)

### 4. Training
You can train the model with the following command:
```
qms_train.py
```
