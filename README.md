# RevisitVLOC

Visual Place Recognition (VPR) tutorial for robot localization using NetVLAD and Milvus vector database.

## Overview

This repository demonstrates how to build a place memory system for indoor robot localization:

1. **NetVLAD** - Convert images to compact 4096-D global descriptors
2. **Milvus Vector DB** - Store and query place memories efficiently
3. **Online Query** - Retrieve similar places for coarse localization

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/RevisitVLOC.git
cd RevisitVLOC
pip install -r requirements.txt
```

### 2. Download Sample Data

Download from [NYC-Indoor-VPR](#acknowledgements) and organize as:

```
data/library/
├── 0407_0415_b4/   # reference
└── 0625_0628_b4/   # query
```

### 3. Run the Notebook

```bash
jupyter notebook pipeline_InLoc_indoor.ipynb
```

## Project Structure

```
RevisitVLOC/
├── pipeline_InLoc_indoor.ipynb  # Main tutorial notebook
├── vpr_core/                    # Feature extraction modules
│   └── extractors/              # NetVLAD, SuperPoint, etc.
├── utils/
│   └── vector_DB_IO.py          # Milvus database manager
└── data/                        # Sample datasets (download separately)
```

## Requirements

- Python 3.12
- PyTorch
- Milvus Lite (pymilvus)
- OpenCV, h5py, numpy, matplotlib

## Dataset

The sample dataset contains indoor **library** images

## Acknowledgements

This project builds upon the following excellent works:

- **[NetVLAD](https://github.com/Relja/netvlad)** - CNN architecture for place recognition
  > Arandjelović et al. "NetVLAD: CNN architecture for weakly supervised place recognition." CVPR 2016.

- **[hloc](https://github.com/cvg/Hierarchical-Localization)** - Hierarchical localization toolbox
  > Sarlin et al. "From Coarse to Fine: Robust Hierarchical Localization at Large Scale." CVPR 2019.

- **[NYC-Indoor-VPR](https://ai4ce.github.io/NYC-Indoor-VPR/#)** - Indoor VPR dataset  
  > [huggingface](https://huggingface.co/datasets/ai4ce/NYC-Indoor-VPR-Data/tree/main) Download public dataset  
  > Diwei Sheng et al. "NYC-Indoor-VPR: A Long-Term Indoor Visual Place Recognition Dataset." 2024.

## License

Apache 2.0
