The code in this folder samples the guardian paragraphs that will be used for annotation with the objective of training an EL model.

The notebooks should be run sequentially according to the numbering on their file name. 

1_stratified_url_sample.ipynb - Contains code to gather Guardian articles from 2010 to 2022 and perform stratified sampling of a subset pool of Guardian content

2_spacy_trf_model_ner_all_files.ipynb - Download stratified content from S3 buck and use spaCy's en_core_web_trf NER model to extract mentioned entities 

3_gu_paragraph_sample.ipynb - Downsample paragraphs to use in the creation of the final dataset to train/

All datasets imported by these notebooks should be stored under /1_sample_gu_paragraphs/assets/