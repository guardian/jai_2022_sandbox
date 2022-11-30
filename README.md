# Training an Entity Linker model using spaCy and Prodigy 

This repository was created in December 2022 as a result of a project undertaken for the Journalism AI 2022 Fellowship. 
The motivation behind the project can be found in this [Notion page](https://www.notion.so/badwillhunting/Recognising-bad-actors-in-data-leaks-with-AI-19d40278356f4f3eb52d5d1678d14971)

The project was conducted by:
    - Alet Law and Tinashe Munyuki from the [Daily Maverick](https://www.dailymaverick.co.za/)
    - Dimitri Tokmetzis and Heleen Emanuel from [Follow the Money](https://www.ftm.eu/)
    - Luis Flores, Michel Schammel and Anna Vissen from [The Guardian](https://www.theguardian.com/), the main developers of the codebase

This Python repository provides a template of the code required to train a [spaCy](https://spacy.io/) entity linker model for PERSON text mentions. 

## Structure:

/src - code to pre-process datasets and train an EL model

/prodigy - custom recipe to annotate training/test datasets 

/data - dataset directory 


## Datasets: 

### Knowledge Bases 
Our template uses two open-access Knowledge Bases (KBs) of potential interest to investigative reporting, 
namely [Open Sanctions](https://www.opensanctions.org/) and [LittleSis](https://littlesis.org/home/dashboard).
The complete datasets can be freely obtained from either source under json format.

The template includes pre-processing steps for each KB to transform the raw json data into pandas dataframes. 
Each entity in the processed dataset will have an individual ID, an alias and a description field.
Please be aware the final dataset still contains duplicated entities, i.e. two or more IDs referring to the same "real-world" entity. 

### Text Documents 
The documents originally used in this work were article paragraphs acquired through the [Guardian Content API](https://open-platform.theguardian.com/)
This dataset is interchangeable with any structured text containing named entities that be linked to the KBs. 

### Prodigy annotations 
The training and test datasets used to train and evaluate the model were manually generated using [Prodigy](https://prodi.gy/). 
The custom Prodigy recipe used in this task is included 

## Running the code:

The code requires pre-installation of the [DVC](https://dvc.org/) data versioning package in the project environment.
Running the command `dvc repro` will run the code end-to-end. 

## Packaging the model:

`python -m spacy package --force <trained_model_input_> <compiled_model_output_directory> --code scripts/.py`
[More information](https://spacy.io/api/cli#package)



