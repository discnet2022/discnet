# Reproduce experimental resualts on the full setting

- Install dependencies:
  - python 3.8.0
  - torch 1.8.0
  - spacy 3.0.6
  - pandas 1.2.3
  - numpy 1.21.2
  - tqdm
- Place `NOTEEVENTS.csv` from MIMIC-III dataset in  `mimic3_data` folder.
- Open `preprocessing.ipynb` with Jypyter Notebook and run all cells. Due to slow SpaCy lemmatization, it takes about 6-8 hours to finish preprocessing.
- Unzip `word_embeddings.npy.zip` in `preprocessed_mimic3` folder.
- run `train_discnet.py`, It takes about 36-48 hours to finish training on a GeForce RTX 2080 Ti.

# Potential issues

- Only support single GPU training.
- There may be potential bugs. We are still polishing our code. 

