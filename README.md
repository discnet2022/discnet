# Reproduce the experimental results on the MIMIC-III full setting.

- Install dependencies:
  - python 3.8.0
  - torch 1.8.0
  - spacy 3.0.6
  - pandas 1.2.3
  - numpy 1.21.2
  - tqdm
- Place `NOTEEVENTS.csv` from MIMIC-III dataset in  `mimic3_data` folder.
- Open `preprocessing.ipynb` with Jypyter Notebook and run all cells. Due to slow SpaCy lemmatization, it takes about 6-8 hours to finish preprocessing.
- If you are having trouble running preprocessing or if it takes forever to finish preprocessing on your platform, please contact me at [yangwanchun82@gmail.com](mailto:yangwanchun82@gmail.com). Please provide proof that you have worked with the MIMIC dataset or send me your CITI certificate, so that I can send you the download link for the preprocessed dataset for training within 48 hours.
- Unzip `word_embeddings.npy.zip` in `preprocessed_mimic3` folder.
- run `train_discnet.py`, It takes about 36-48 hours to finish training on a GeForce RTX 2080 Ti.
- The training log and the testing results are in the "output_mimic_discnet" folder.
- The trained model can be downloaded from: https://drive.google.com/file/d/1BfDTXWzAvb7p9tkPkjLUV6ixcQp57LjN/view?usp=sharing

# Potential issues

- Only support single GPU training.
- There may be potential bugs. We are still polishing our code. 

