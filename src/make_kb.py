import numpy as np
import pandas as pd
import spacy
from spacy.kb import KnowledgeBase
from pathlib import Path


def load_entities(kb_data):
    """ Create dictionary"""
    names = dict()
    descriptions = dict()

    for row in kb_data.iterrows():
        qid = str(row[1][0])
        name = str(row[1][1])
        desc = str(row[1][2])
        names[qid] = name
        descriptions[qid] = desc

    return names, descriptions

def get_embedding_dim(nlp):
    if 'tok2vec' in nlp.config['components']:
        return nlp.config['components']['tok2vec']['model']['encode']['width']
    else:
        raise NotImplementedError("Implement transformer embeddimg width code")

def main(input_file, output_file, nlp_model, empty=True):

    # Load data and model
    data = pd.read_csv(input_file,index_col=0)
    kb_data=data[['id','name','desc']]
    name_dict, desc_dict = load_entities(kb_data)

    nlp = spacy.load(nlp_model, exclude=['lemmatizer', 'tagger', 'ner'])
    embedding_dims = get_embedding_dim(nlp)

    ## Generate synthetic aliases
    aliases_data = kb_data[kb_data['name'].duplicated(keep=False)].sort_values(['name'])
    aliases_data['id'] = aliases_data['id'].astype(str)

    alias_dict = dict((alias, list(aliases_data.loc[aliases_data['name'] == alias, 'id'].values))
                       for alias in aliases_data['name'].unique())

    # Set up empty KB object and fill with entities
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=embedding_dims)

    ## Add ents to KB
    descriptions_enc = dict()
    if empty:
        for qid, desc in desc_dict.items():
            desc_enc = np.zeros(embedding_dims)
            kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)  # 342 is an arbitrary value here
    else:
        descriptions = desc_dict.values()
        qids = desc_dict.keys()
        for qid, desc_doc in zip(qids, nlp.pipe(descriptions)):
            desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   # 342 is an arbitrary value here



    for qid, name in name_dict.items():
        if name not in alias_dict.keys():
            kb.add_alias(alias=str(name), entities=[str(qid)], probabilities=[1])   # 100% prior probability P(entity|alias)

    ### REFACTOR?
    for alias_ in alias_dict.keys():
        qids=alias_dict[alias_]
        probs = [round(1/len(qids),2)-.01 for qid in qids]
        kb.add_alias(alias=alias_, entities=qids, probabilities=probs)  # sum([probs]) should be <= 1 !


    # Save file
    kb.to_disk(output_file)





if __name__ == '__main__':
    import argparse
    output_file, nlp_model = None, None
    verbose = False

    parser = argparse.ArgumentParser(
        prog='make_kb',
        description='Create knowledge base from KB dataset.'
    )
    parser.add_argument("input_file", default=["data/kb_entities_full.csv"])
    parser.add_argument('-o', '--out', action="store_const", const=output_file, help="Name of output file.",
                        default="kb")  # option that takes a value
    parser.add_argument('-n', '--nlp', action="store_const", const=nlp_model, help="Name of the Spacy model",
                        default="en_core_web_lg")
    parser.add_argument('-e', '--empty', action='store_true', help="Fill embeddings with null vectors")

    args = parser.parse_args()
    main(args.input_files, args.out, args.nlp_model, args.empty)