"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
"""

import string
from operator import itemgetter
from pathlib import Path
from typing import Iterator

import nltk
import pandas as pd
import prodigy
import rapidfuzz
import spacy
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm
from prodigy.components.filters import filter_duplicates
from prodigy.components.loaders import TXT
from prodigy.models.ner import EntityRecognizer
from prodigy.util import set_hashes
from spacy.kb import KnowledgeBase, Candidate  # , get_candidates

from gu_model.trf_tensor_to_vec import *


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
    nlp = spacy.load('gu_model/en_ner_guardian-1.0.3/en_ner_guardian/en_ner_guardian-1.0.3',
                     disable=['transformer', 'tagger', 'parser', 'lemmatizer', 'attribute_ruler'])
    nlp.add_pipe('tensor2attr')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    kb_entities = pd.read_csv(entity_loc, index_col=0)
    kb_entities['id'] = kb_entities['id'].astype(str)
    kb_entities['name'] = kb_entities['name'].astype(str)
    kb_entities['desc'] = kb_entities['desc'].astype(str)
    kb_entities['kb_url'] = kb_entities['kb_url'].astype(str)
    kb_entities = kb_entities[['id', 'name', 'desc', 'kb_url']]

    kb_entities['desc_len'] = kb_entities['desc'].str.len()

    id_dict = dict()
    for row in kb_entities.iterrows():
        qid = str(row[1][0])
        name = str(row[1][1])
        desc = str(row[1][2])
        id_dict[qid] = (name, desc)

    # Initialize the Prodigy stream by running the NER model
    source = pd.read_csv(source, index_col=0)
    stream = TXT(source['paragraphs'].values)
    stream_url = list(source['url'].values)
    stream = [set_hashes(eg) for eg in stream]
    # add gu_url to hashed txt stream
    for dict_, url in zip(stream, stream_url):
        dict_['gu_url'] = url
    stream = (eg for score, eg in model(stream))

    # For each NER mention, add the candidates from the KB to the annotation task
    stream = _add_options(stream, kb, nlp, id_dict, kb_entities)  # _url)
    stream = filter_duplicates(stream, by_input=False, by_task=True)

    blocks = [{"view_id": "html",
               "html_template": "<a href="'https://{{gu_url}}' + " target='_blank'>Guardian Article URL</a>"
               },
              {"view_id": "choice"},
              ]

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {"blocks": blocks,
                   "choice_auto_accept": False,
                   "buttons": ["accept", "undo"],
                   }
    }


def get_candidates_from_fuzzy_matching(span, kb, single_name=False, matching_thres=60) -> Iterator[Candidate]:
    """
    Return a list of candidate entities for an alias based on fuzzy string matching.
    Each candidate defines the entity, the original alias,
    and the prior probability of that alias resolving to that entity.
    If the alias is not known in the KB, and empty list is returned.
    """
    aliases = kb.get_alias_strings()
    matches = {}
    for al in aliases:
        if single_name:
            # For single name mentions, e.g. Trump, use partial_ratio
            fuzzy_ratio = rapidfuzz.fuzz.partial_ratio(span.lower(), al.lower())
        else:
            # For multi name mentions, e.g. Donald Trump, use WRatio
            fuzzy_ratio = rapidfuzz.fuzz.WRatio(span.lower(), al.lower())
        if fuzzy_ratio >= matching_thres:
            matches[al] = fuzzy_ratio
    candidates = []
    for match in matches:
        candidates.extend(kb.get_alias_candidates(match))
    return candidates, matches

def relevant_lexicon(text, stopwords):
    # Extract the set of words from a text field
    punctuation_rm = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_rm)
    word_set = set(word_tokenize(text))
    return word_set.difference(stopwords)


def _add_options(stream, kb, nlp, id_dict, kb_entities):  # _url):
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
                mention = span.text
                single_name = len(mention.split(' ')) <= 1
                # Retrieve a wide list of candidates based on fuzzy string matching and similarity score
                candidates, matches = get_candidates_from_fuzzy_matching(mention, kb, single_name)

                # Convert candidate and scores to df
                matches_df = pd.DataFrame.from_dict(matches, orient='index').reset_index().rename(
                    columns={'index': 'name', 0: 'score'})
                # Merge kb entity description
                matches_df = matches_df.merge(kb_entities, on=['name'], how='left')
                # Retain candidates with scores in top decile
                top_decile=0.9
                score_thres = matches_df['score'].quantile(top_decile)
                matches_df = matches_df[matches_df['score'] >= score_thres]
                # Sort based on score and length of description text
                matches_df = matches_df.sort_values(by=['score', 'desc_len'], ascending=False).reset_index(drop=True)
                # Increase score in candidates with a wikiID
                matches_df.loc[matches_df['id'].str.match('Q\d'), 'score'] = matches_df.loc[matches_df['id'].str.match(
                    'Q\d'), 'score'] * 1.05
                # Calculate normalised min/max description length.
                # Descriptions over 300 characters long are capped at 1.
                max_len = 300
                min_len = matches_df['desc_len'].min()
                matches_df['normalised_desc_len'] = (matches_df['desc_len'] - min_len) / (max_len - min_len)
                matches_df['normalised_desc_len'] = matches_df['normalised_desc_len'].apply(lambda x: 1 if x > 1 else x)

                # Find common words between paragraph and candidate descriptions
                stopwords = set(nltk.corpus.stopwords.words())
                desc_ents_d = dict()
                column_order = matches_df.reset_index().columns.values
                id_loc, = np.where(column_order == 'id')
                desc_loc, = np.where(column_order == 'desc')
                for tuple_ in matches_df.itertuples():
                    qid = tuple_[id_loc[0]]
                    desc = tuple_[desc_loc[0]]
                    desc_ents = relevant_lexicon(desc, stopwords)
                    desc_ents_d[qid] = desc_ents

                common_ent_count_d = {}
                sentence_ents = [str(ent) for ent in nlp(text).ents]
                sentence_ents = set(' '.join(sentence_ents).split())
                for qid, ent in desc_ents_d.items():
                    common_ent_counts = len(sentence_ents.intersection(ent))
                    common_ent_count_d[qid] = common_ent_counts

                common_ent_counts = pd.DataFrame.from_dict(common_ent_count_d, orient='index').reset_index().rename(
                    columns={'index': 'id', 0: 'common_ent_counts'})

                matches_df = matches_df.merge(common_ent_counts, on='id')

                matches_df = matches_df.sort_values(by=['common_ent_counts', 'score', 'normalised_desc_len'],
                                                    ascending=False).reset_index(drop=True)

                n_candidates = 10
                matches_df = matches_df.head(n_candidates)

                candidates = matches_df['id'].map({candidate.entity_: candidate for candidate in candidates}).to_list()
                matches_df = dict(zip(matches_df['id'], matches_df['score']))

                if candidates:
                    options = []
                    # add in a few additional options in case a correct ID cannot be picked
                    options.append({"id": "NER_WrongType", "text": "Incorrect entity type returned by NER model."})
                    options.append({"id": "NEL_MoreContext", "text": "Need more context to decide."})
                    options.append({"id": "NEL_NoCandidate", "text": "No viable candidate."})

                    options.extend([
                        {"id": c.entity_, "html": _print_info(c.entity_, id_dict, matches[c.alias_], kb_entities)}
                        # _url)}
                        for c in candidates
                    ])

                    task["options"] = options
                    task["config"] = {"choice_style": "multiple"}
                    yield task


def shorten_description(descr, n_splits=2, min_desc_len=50, max_desc_len=500):
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


def _print_info(entity_id, id_dict, score, kb_entities):  # _url):
    """For each candidate QID, create a link to the corresponding Wikidata page and print the description"""
    name, descr = id_dict.get(entity_id)
    # score=round(score)
    # url = kb_entities_url.loc[kb_entities_url['id'] == str(entity_id), 'kb_url'].values[0]
    url = kb_entities.loc[kb_entities['id'] == str(entity_id), 'kb_url'].values[0]
    # Tailor output description length
    short_descr = shorten_description(descr)
    # Prodigy option display text
    option = "<a class=\"entLink\" href='" + f'{url}' + "' target='_blank' onclick=\"clickMe(event)\">" + entity_id + "</a>"
    option += ": " + name + '; \n' + short_descr
    return option


### Not used in current iteration
### Here for further development

def embed_text(text, nlp):
    """
    Return spaCy embedding of a text.
    """
    return nlp(text).vector


def get_all_kb_candidates(kb):
    """
    Retrieve all candidates from kb.
    """
    aliases = kb.get_alias_strings()
    candidates = []
    for alias in aliases:
        candidates.extend(kb.get_alias_candidates(alias))
    return candidates


def calculate_cosine_similarity(descriptions_vec, vector_ref_sentence):
    """
    Return a dictionary mapping the kb entity id to cosine similarity score
    between kb embedded descriptions and the reference vector.
    """
    similarity = {}
    for entity_id in descriptions_vec.keys():
        vector_desc = descriptions_vec[entity_id]
        score = np.nan_to_num(
            dot(vector_ref_sentence, vector_desc) /
            (norm(vector_ref_sentence) * norm(vector_desc))
            , 0)
        similarity[entity_id] = score
    return similarity


def get_candidates_from_context(text, nlp, candidates, matches, candidate_limit=20):
    """
    Select only the top candidates to surface via the Prodigy UI. Based on
    topmost cosine similarities.
    """
    vector_ref_sentence = embed_text(text, nlp)
    names = dict()
    descriptions_vec = dict()
    for candidate in candidates:
        qid = candidate.entity_
        name = candidate.alias_
        desc_enc = candidate.entity_vector
        names[qid] = name
        descriptions_vec[qid] = desc_enc

    similarity = calculate_cosine_similarity(descriptions_vec, vector_ref_sentence)
    fuzzy_scores = {qid: matches[alias] for qid, alias in names.items()}
    qids = set(similarity.keys()) | set(fuzzy_scores.keys())
    fuzzy_similarity = {}
    for qid in qids:
        fuzzy_similarity[qid] = np.ma.average([similarity[qid], fuzzy_scores[qid] / 100], weights=[1, 3])
    top_best_candidates = dict(sorted(fuzzy_similarity.items(), key=itemgetter(1), reverse=True)[:candidate_limit])
    selected_candidates = [candidate for candidate in candidates if candidate.entity_ in top_best_candidates.keys()]
    return selected_candidates


def order_candidates_alphabetically(candidates):
    """
    Order candidate list alphabetically
    """
    candidates_alphabetical = {candidate.alias_ + ' ' + candidate.entity_: candidate for candidate in candidates}
    candidates_alphabetical = dict(sorted(candidates_alphabetical.items(), key=itemgetter(0), reverse=False))
    return [candidate for candidate in candidates_alphabetical.values()]
