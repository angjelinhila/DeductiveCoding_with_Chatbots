# Deductive Coding with Chatbots

The overarching goal of this project is to provide replicable pipelines for looping LLM Chatbots into qualitative coding tasks. We're currently working on validating deductive coding intervensions that yield accurate results and meet interrate reliability benchmarks. 
In deductive coding, the taxonomy is specified prior to the dataset. We have chose the Comparative Agendas Codebook as our baseline model since it has wide policy applicability and has been validated to apply interculturally. 

In inductive coding, the codebook emerges from the dataset, a posteriori. The end-goal of this projec to contribute novel methods that loop automated coding with Chatbots in a way that augments the task of human coders. The chief advantage is to achieve scalability for large datasets. 

We provide a pipeline for validating discriminant and covergent validity across four intervention methods for Chatbot deductive classification tasks.

30 random samples were drawm from Comparative Agendas Supreme_Court_Cases file. The original file contains 10237 rows, each corresponding to a US Supreme Court Case. 
Each case is labelled with a two-tier hierarchical class: majortopic and subtopic. Original labels were integer keys to the Comparative Agendas Codebook. 

We converted each integer class into a categorical class, labelled: major_label and sub_label. The dataset was consequently randomized using the Hashlib module via the RSA's MD5 algorithm. 
From this randomized population set we drew 30 random samples sets of 30 observations each. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
2. Navigate into directory:
   ```bash
   cd your-repo
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Configure
   Update config.yaml as needed.  

## Usage
1. To run the classification script:
```bash
   python scripts/classification.py
```
2. To run the evaluation script:
```bash
   python scripts/evaluation.py 
```

## Repository Structure
   ```bash
   tree -L 2
   your-repo/
   ├── data/
   │   ├── input_files/
   │   └── processed_files/
   ├── models/
   │   └── classification_model.pkl
   ├── scripts/
   │   ├── classification.py
   │   └── evaluation.py
   ├── config.yaml
   ├── requirements.txt
   └── README.md
   ```

## Contributions
Contributions are welcome! Please see CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Contact 
For questions or suggestions, please contact ahila@utexas.com.
