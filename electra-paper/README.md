
This is codes for our experment that is run by Electra
# Run Experiment
Plesae run run_paper_cs.ipynb for Wang dataset and run_paper_springer.ipynb for the Springer dataset which we proposed.
```
FEATURES = ["title", "keywords", "abstract"] # ["title"]
if "abstract" in FEATURES: max_seq_length = 512
elif len(FEATURES) == 2: max_seq_length = 256
else: max_seq_length = 128
fff = ",".join(FEATURES)
DATA_DIR = "../data/cs-paper/"
OUTPUT_DIR = "gs://paper/xlnet-large-cs-{}/".format("+".join(FEATURES))
PRETRAIN_MODEL_DIR = "gs://bert-eng/xlnet-large-cased"
```
Select feature which you want to run experiment with Feature. DATA_DIR is the location of dataset that you have defined. OUTPUT_DIR is the location of the result (.tsv) that you have defined. PRETRAIN_MODEL_DIR is the location of pretrain model.

# Run Prediction
To predict, run file run_predict.ipynb for predicting Springer dataset and run_predict_cs.ipynb for predicting Wang dataset


