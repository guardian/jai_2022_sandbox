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
from prodigy.components.loaders import TXT, JSONL
from prodigy.util import set_hashes, log
from prodigy.components.filters import filter_duplicates

from pathlib import Path

import rapidfuzz

import pandas as pd
from operator import itemgetter

@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_loc=(
        "Path to OR name of the NLP model with a pretrained NER component",
        "positional",
        None,
        str,
    ),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=(
        "Path to the file with additional information about the entities",
        "positional",
        None,
        Path,
    ),
)

def entity_linker_manual(dataset, source, nlp_loc, kb_loc, entity_loc):
    # Load the NLP and KB objects from file
    nlp = spacy.load(nlp_loc, exclude=["tagger", "parser", "lemmatizer", "attribute_ruler", "tok2vec", "transformer"])
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    kb_entities = pd.read_csv(entity_loc, index_col=0)
    kb_entities['id'] = kb_entities['id'].astype(str)
    kb_entities['name'] = kb_entities['name'].astype(str)
    kb_entities['desc'] = kb_entities['desc'].astype(str)
    kb_entities['kb_url'] = kb_entities['kb_url'].astype(str)
    kb_entities_url = kb_entities[['id', 'name', 'kb_url']]
    kb_entities = kb_entities[['id', 'name', 'desc']]
    id_dict = dict()
    for row in kb_entities.iterrows():
        qid = str(row[1][0])
        name = str(row[1][1])
        desc = str(row[1][2])
        id_dict[qid] = (name, desc)
    del(kb_entities)
    # Initialize the Prodigy stream by running the NER model
    ## Faster reading by using generators
    stream = JSONL(source)
    stream = (set_hashes(eg) for eg in stream)
    stream = (eg for score, eg in model(stream))

    # For each NER mention, add the candidates from the KB to the annotation task as well as the url of the article
    stream = _add_options(stream, kb, id_dict, kb_entities_url)
    stream = filter_duplicates(stream, by_input=False, by_task=True)

    blocks=[{"view_id": "html",
             "html_template": "<a href=https://{{url}} target=\"_blank\">Guardian Article URL</a>"
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
    matches={}
    for al in aliases:
        fuzzy_ratio = rapidfuzz.fuzz.WRatio(span.lower(), al.lower())
        if fuzzy_ratio >=matching_thres:
            matches[al]=fuzzy_ratio
    candidates=[]
    for match in matches:
        candidates.extend(kb.get_alias_candidates(match))
    return candidates, matches

def order_candidates_fuzzy_score(candidates, matches, candidate_limit=12):
    """
    Order candidates by descending fuzzy name matching score
    """
    candidate_d = dict()
    fuzzy_scores = dict()
    for candidate in candidates:
        qid = candidate.entity_
        name = candidate.alias_
        candidate_d[qid] = candidate
        fuzzy_scores[qid] = matches[name]
    entities_ordered = dict(sorted(fuzzy_scores.items(), key=itemgetter(1), reverse=True))
    entities_ordered = list(entities_ordered.keys())[:candidate_limit]
    return [candidate_d[entity] for entity in entities_ordered]

def _add_options(stream,  kb, id_dict, kb_entities_url):
    """Define the options the annotator will be given, by consulting the candidates from the KB for each NER span
    using a bespoke logic to only surface plausibly relevant candidates.
    """
    for task in stream:
        for mention in filter(lambda m: m["label"] in ["PERSON"], task["spans"]):
            start_char = int(mention["start"])
            end_char = int(mention["end"])
            span_text = task["text"][start_char: end_char]
            candidates, matches = get_candidates_from_fuzzy_matching(span_text, kb)

            candidates=order_candidates_fuzzy_score(candidates, matches)
            if candidates:
                options=[]
                # we add in a few additional options in case a correct ID can not be picked
                options.append({"id": "NER_WrongType", "text": "Incorrect entity type returned by NER model."})
                options.append({"id": "NEL_MoreContext", "text": "Need more context to decide."})
                options.append({"id": "NEL_NoCandidate", "text": "No viable candidate."})

                options.extend([
                    {"id": c.entity_, "html": _print_info(c.entity_, id_dict, matches[c.alias_], kb_entities_url)}
                    for c in candidates
                ])

                task["options"] = options
                task["config"] = {"choice_style": "multiple"}
                yield task

def shorten_description(descr, n_splits = 2, min_desc_len = 50, max_desc_len = 500):
    # Keep description output len within min and max limits
    # This only affects the Prodigy UI text, not the original KB desc.
    short_descr = '. '.join(descr.split('. ')[:n_splits])
    while len(short_descr) < min_desc_len:
        # Keep extending description length
        n_splits += 1
        new_desc_str = '. '.join(descr.split('. ')[:n_splits])
        if len(new_desc_str) == len(short_descr):
            # Max possible len break clause
            break
        short_descr = new_desc_str
        if len(short_descr) > max_desc_len:
            # Keep description length up to max limit
            n_splits -= 1
            short_descr = '. '.join(descr.split('. ')[:n_splits])
            break
    if short_descr.replace(' ', '')[-1] != '.':
        # Ensure stop mark at the end of description
        short_descr += '.'
    return short_descr

def _print_info(entity_id, id_dict, score, kb_entities_url):
    """For each candidate QID, create a link to the corresponding Wikidata page and print the description"""
    name, descr = id_dict.get(entity_id)
    # score=round(score)
    url = kb_entities_url.loc[kb_entities_url['id'] == str(entity_id), 'kb_url'].values[0]
    # Tailor output description length
    short_descr = shorten_description(descr)
    # Prodigy option display text
    option = "<a class=\"entLink\" href='" + f'{url}' + "' target='_blank' onclick=\"clickMe(event)\">" + entity_id + "</a>"
    option += ": " + name + '; \n' + short_descr
    return option