from typing import Iterator
from spacy.kb import KnowledgeBase, Candidate
import pdb

@profile
def get_custom_candidates(kb: KnowledgeBase, span) -> Iterator[Candidate]:
    """
    Return candidate entities for a given span by using the text of the span as the alias
    and fetching appropriate entries from the index.
    This function is also checking if the input span text is a single word and matches any occurences of that word
    in the knowledge base to surface additional candidates.
    """
    ##
    # Get candidates based on whole span text
    candidates = kb.get_alias_candidates(span.text)
    if len(span.text.split()) == 1:
        # Only have a single name in span (assuming it's a surname),
        # get other variations of names in the KB that partially match the span text
        other_options = filter(lambda ent: span.text in ent, kb.get_alias_strings())
        other_candidates = tuple(tuple(kb.get_alias_candidates(option)) for option in other_options)
        pdb.set_trace()
        # for option in other_options:
        #     candidates.extend(kb.get_alias_candidates(option))
        # pdb.set_trace()
        return candidates + list(other_candidates)
    return candidates
