The code in this folder imports data from two public access Knowledge Bases, namely Open Sanctions and LittleSis, containing information about real world players involved in politics, business and crime.

The notebooks process the data from each KB separately and then append them across common columns to form a combined dataset. This final dataset will be converted into a spaCy KB object to be used by an EL model.

The notebooks should be run sequentially according to the numbering on their file name. 

1_preprocess_open_sanctions.ipynb - Processes the Open Sanctions dataset

2_preprocess_little_sis.ipynb - Processes the LittleSis dataset

3_merge_and_clean_kb_datasets.ipynb - Joins the two datasets together into a single file

4_create_spacy_KB - Generate a spaCy KB object from the dataset and save to disk. 
The object requires three fields, a unique entity ID, an alias, and a pre-embedded description.

All datasets imported by these notebooks should be stored under /2_kb_datasets/assets/