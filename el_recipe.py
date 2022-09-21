"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
You can run this project without having Prodigy or using this recipe:
sample results are stored in assets/emerson_annotated_text.jsonl
"""

from typing import Iterator

import spacy
from spacy.kb import KnowledgeBase, Candidate #, get_candidates

import prodigy
from prodigy.models.ner import EntityRecognizer
from prodigy.components.loaders import TXT
from prodigy.util import set_hashes
from prodigy.components.filters import filter_duplicates

import csv
from pathlib import Path

import rapidfuzz


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
)
def entity_linker_manual(dataset, source, nlp_dir, kb_loc, entity_loc):
    # Load the NLP and KB objects from file
    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    id_dict = dict()
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2])

    # Initialize the Prodigy stream by running the NER model
    stream = TXT(source)


    ## lookup batch size

    stream = [set_hashes(eg) for eg in stream]
    stream = (eg for score, eg in model(stream))

    # For each NER mention, add the candidates from the KB to the annotation task
    stream = _add_options(stream, kb, nlp, id_dict)
    stream = filter_duplicates(stream, by_input=False, by_task=True)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {"choice_auto_accept": False},
    }


def get_fuzzy_matching(span, kb) -> Iterator[Candidate]:
    """
    Return candidate entities for an alias. Each candidate defines the entity, the original alias,
    and the prior probability of that alias resolving to that entity.
    If the alias is not known in the KB, and empty list is returned.
    """

    aliases=kb.get_alias_strings()
    matches=[]
    for al in aliases:
        fuzzy_ratio=rapidfuzz.fuzz.token_set_ratio(span,al)
        if fuzzy_ratio >=99:
            matches.append(al)

    candidates=[]
    for match in matches:
        candidates.extend(kb.get_alias_candidates(match))

    return candidates

def _add_options(stream, kb, nlp, id_dict):
    """Define the options the annotator will be given, by consulting the candidates from the KB for each NER span."""
    for task in stream:
        text = task["text"]
        for mention in task["spans"]:
            if mention["label"] in ['PERSON','ORG']:
                start_char = int(mention["start"])
                end_char = int(mention["end"])
                doc = nlp(text)
                span = doc.char_span(start_char, end_char, mention["label"])

                #candidates = kb.get_alias_candidates(span.text)
                candidates= get_fuzzy_matching(span.text, kb)
                if candidates:
                    options = [
                        {"id": c.entity_, "html": _print_url(c.entity_, id_dict)}
                        for c in candidates
                    ]

                    # we sort the options by ID
                    #options = sorted(options, key=lambda r: int(r["id"][1:]))

                    # we add in a few additional options in case a correct ID can not be picked
                    options.append({"id": "NIL_otherLink", "text": "Link not in options"})
                    options.append({"id": "NIL_ambiguous", "text": "Need more context"})

                    task["options"] = options
                    yield task


def _print_url(entity_id, id_dict):
    """For each candidate QID, create a link to the corresponding Wikidata page and print the description"""
    name, descr = id_dict.get(entity_id)
    option =  name + " [" + entity_id + "]: " + (descr[:250] if len(descr) >= 150 else descr)
    return option