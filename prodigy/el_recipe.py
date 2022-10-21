"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
You can run this project without having Prodigy or using this recipe:
sample results are stored in assets/emerson_annotated_text.jsonl
"""

from typing import Iterator, Optional

import spacy
from spacy.kb import KnowledgeBase, Candidate #, get_candidates

import prodigy
from prodigy.models.ner import EntityRecognizer
from prodigy.components.loaders import TXT
from prodigy.util import set_hashes, log
from prodigy.components.filters import filter_duplicates

#import csv
from pathlib import Path

import rapidfuzz

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter


@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_dir=(
        "Path to the NLP model with a pretrained NER component",
        "positional",
        None,
        Path,
    ),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=(
        "Path to the file with additional information about the entities",
        "positional",
        None,
        Path,
    ),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str)
)
def entity_linker_manual(dataset, source, nlp_dir, kb_loc, entity_loc):
    # Load the NLP and KB objects from file
    nlp = spacy.load(nlp_dir, exclude=["tagger", "parser", "lemmatizer", "attribute_ruler", "tok2vec"])
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    id_dict = dict()
    
    #kb_entities=pd.read_csv(entity_loc, names=['qid','name','desc'])
    kb_entities = pd.read_csv(entity_loc, index_col=0)
    kb_entities['id']=kb_entities['id'].astype(str)
    kb_entities['name'] = kb_entities['name'].astype(str)
    kb_entities['desc'] = kb_entities['desc'].astype(str)
    kb_entities['kb_url'] = kb_entities['kb_url'].astype(str)
    kb_entities_url=kb_entities[['id','name','kb_url']]
    kb_entities = kb_entities[['id', 'name', 'desc']]
    for row in kb_entities.iterrows():
        qid = str(row[1][0])
        name = str(row[1][1])
        desc = str(row[1][2])
        id_dict[qid] = (name, desc)

    #with entity_loc.open("r", encoding="utf8") as csvfile:
        #csvreader = csv.reader(csvfile, delimiter=",")
        #for row in csvreader:
            #id_dict[row[0]] = (row[1], row[2])

    # Initialize the Prodigy stream by running the NER model
    source=pd.read_csv(source, index_col=0)
    #stream = TXT(source)
    stream=TXT(source['paragraphs'].values)
    stream_url = list(source['url'].values)
    stream = [set_hashes(eg) for eg in stream]
    # add gu_url to hashed txt stream
    for dict_, url in zip(stream, stream_url):
        dict_['gu_url'] = url
    stream = (eg for score, eg in model(stream))


    # For each NER mention, add the candidates from the KB to the annotation task
    stream = _add_options(stream, kb, nlp, id_dict, kb_entities_url)
    stream = filter_duplicates(stream, by_input=False, by_task=True)

    blocks=[{"view_id": "html",
             "html_template": "<a href="'https://{{gu_url}}'+">Guardian Article URL</a>"
         },
            {"view_id":"choice"},
         ]

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {"blocks":blocks,"choice_auto_accept": False, "buttons": ["accept", "undo"],}
    }

def get_candidates_from_fuzzy_matching(span, kb, matching_thres=60) -> Iterator[Candidate]:
    """
    Return a list of candidate entities for an alias based on fuzzy string matching.
    Each candidate defines the entity, the original alias,
    and the prior probability of that alias resolving to that entity.
    If the alias is not known in the KB, and empty list is returned.
    """
    aliases=kb.get_alias_strings()
    #matches=[]
    matches={}
    for al in aliases:
        #fuzzy_ratio=rapidfuzz.fuzz.token_set_ratio(span.lower(),al.lower())
        fuzzy_ratio = rapidfuzz.fuzz.WRatio(span.lower(), al.lower())
        if fuzzy_ratio >=matching_thres:
            #matches.append(al)
            matches[al]=fuzzy_ratio
    candidates=[]
    for match in matches:
        candidates.extend(kb.get_alias_candidates(match))
    return candidates, matches

def order_candidates_fuzzy_score(candidates, matches, candidate_limit=12):
    """
    Order candidates by descending fuzzy name matching score
    """
    # names = dict()
    candidate_d = dict()
    fuzzy_scores = dict()
    for candidate in candidates:
        qid = candidate.entity_
        name = candidate.alias_
        # names[qid] = name
        candidate_d[qid] = candidate
        fuzzy_scores[qid] = matches[name]
    entities_ordered = dict(sorted(fuzzy_scores.items(), key=itemgetter(1), reverse=True))
    entities_ordered = list(entities_ordered.keys())[:candidate_limit]
    return [candidate_d[entity] for entity in entities_ordered]

def _add_options(stream, kb, nlp, id_dict, kb_entities_url):
    """Define the options the annotator will be given, by consulting the candidates from the KB for each NER span
    using a bespoke logic to only surface plausibly relevant candidates.
    """
    for task in stream:
        text = task["text"]
        for mention in task["spans"]:
            if mention["label"] in ['PERSON']:
                start_char = int(mention["start"])
                end_char = int(mention["end"])
                doc = nlp(text)
                span = doc.char_span(start_char, end_char, mention["label"])
                candidates, matches = get_candidates_from_fuzzy_matching(span.text, kb)

                """
                candidate_limit=10
                if not candidates:
                    candidates=get_all_kb_candidates(kb)
                    candidates=get_candidates_from_context(text, nlp, candidates, matches, candidate_limit)

                if len(candidates) > candidate_limit:
                    candidates=get_candidates_from_context(text, nlp, candidates, matches, candidate_limit)
                """

                #candidates=order_candidates_alphabetically(candidates)
                candidates=order_candidates_fuzzy_score(candidates, matches)
                if candidates:
                    options=[]
                    # we add in a few additional options in case a correct ID can not be picked
                    options.append({"id": "NEL_otherLink", "text": "No viable candidate."})
                    options.append({"id": "NEL_ambiguous", "text": "Need more context to decide."})
                    options.append({"id": "NER_mislabeled", "text": "Incorrect entity type returned by NER model."})

                    options.extend([
                        {"id": c.entity_, "html": _print_info(c.entity_, id_dict, matches[c.alias_], kb_entities_url)}
                        for c in candidates
                    ])

                    # we sort the options by ID
                    #options = sorted(options, key=lambda r: int(r["id"][1:]))

                    task["options"] = options
                    task["config"] = {"choice_style": "multiple"}
                    yield task

def _print_info(entity_id, id_dict, score, kb_entities_url):
    """For each candidate QID, create a link to the corresponding Wikidata page and print the description"""
    name, descr = id_dict.get(entity_id)
    score=round(score)
    url=kb_entities_url.loc[kb_entities_url['id']==str(entity_id), 'kb_url'].values[0]
    descr_two_sentences='.'.join(descr.split('.')[:2])+'.'
    option = "<a class=\"entLink\" href='" + f'{url}' + "' target='_blank' onclick=\"clickMe(event)\">" + entity_id + "</a>"
    option += ": " + name + '; ' + descr_two_sentences
    return option


###
def embed_text(text,nlp):
    """
    Return spaCy embedding of a text.
    """
    return nlp(text).vector

def get_all_kb_candidates(kb):
    """
    Retrieve all candidates from kb.
    """
    aliases=kb.get_alias_strings()
    candidates=[]
    for alias in aliases:
        candidates.extend(kb.get_alias_candidates(alias))
    return candidates

def calculate_cosine_similarity(descriptions_vec,vector_ref_sentence):
    """
    Return a dictionary mapping the kb entity id to cosine similarity score
    between kb embedded descriptions and the reference vector.
    """
    similarity={}
    for entity_id in descriptions_vec.keys():
        vector_desc=descriptions_vec[entity_id]
        score=np.nan_to_num(
            dot(vector_ref_sentence, vector_desc)/
            (norm(vector_ref_sentence)*norm(vector_desc))
        ,0)
        similarity[entity_id]=score
    return similarity

def get_candidates_from_context(text, nlp, candidates, matches, candidate_limit=20):
    """
    Select only the top candidates to surface via the Prodigy UI. Based on
    topmost cosine similarities.
    """
    vector_ref_sentence=embed_text(text,nlp)
    names = dict()
    descriptions_vec = dict()
    for candidate in candidates:
        qid = candidate.entity_
        name = candidate.alias_
        desc_enc = candidate.entity_vector
        names[qid] = name
        descriptions_vec[qid] = desc_enc

    similarity=calculate_cosine_similarity(descriptions_vec, vector_ref_sentence)
    fuzzy_scores = {qid: matches[alias] for qid, alias in names.items()}
    qids = set(similarity.keys()) | set(fuzzy_scores.keys())
    fuzzy_similarity = {}
    for qid in qids:
        fuzzy_similarity[qid] = np.ma.average([similarity[qid], fuzzy_scores[qid] / 100], weights=[1, 3])
    top_best_candidates = dict(sorted(fuzzy_similarity.items(), key = itemgetter(1), reverse = True)[:candidate_limit])
    selected_candidates = [candidate for candidate in candidates if candidate.entity_ in top_best_candidates.keys()]
    return selected_candidates

def order_candidates_alphabetically(candidates):
    """
    Order candidate list alphabetically
    """
    candidates_alphabetical = {candidate.alias_ + ' ' + candidate.entity_: candidate for candidate in candidates}
    candidates_alphabetical = dict(sorted(candidates_alphabetical.items(), key=itemgetter(0), reverse=False))
    return [candidate for candidate in candidates_alphabetical.values()]