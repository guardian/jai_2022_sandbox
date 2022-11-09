import pandas
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin, Span

def clean_data(df):
    """ Implement data cleaning. """
    return df


def make_doc(example, nlp):
    """ Make spacy document from prodigy example (dict)"""
    sentence = example["text"]
    gold_ids = []
    if example["answer"] == "accept":
        QID = example["accept"][0]
        doc = nlp.make_doc(sentence)
        gold_ids.append(QID)
        # we assume only 1 annotated span per sentence, and only 1 KB ID per span
        entity = doc.char_span(
            example["spans"][0]["start"],
            example["spans"][0]["end"],
            label=example["spans"][0]["label"],
            kb_id=QID,
        )
        doc.ents = [entity]
        for i, t in enumerate(doc):
            doc[i].is_sent_start = i == 0
        return doc

def main(input_files, out_stem, nlp_model = 'en_core_web_lg', verbose=False):
    inputs = [
        pandas.read_json(inp, lines=True)
        for inp in input_files
    ]
    df = pandas.concat(inputs)

    # Clean data
    df = clean_data(df)

    # Split datasets and ensure no paragraph is split between train and dev
    index_train, index_test = train_test_split(df['_input_hash'].unique(), test_size=0.4, random_state=14)

    df_train = df[df['_input_hash'].isin(index_train)]
    df_test = df[df['_input_hash'].isin(index_test)]

    # Make docs
    nlp = spacy.load(nlp_model, exclude="parser, tagger")
    train_docs = df_train.apply(make_doc, args=(nlp,), axis=1)
    test_docs = df_test.apply(make_doc, args=(nlp,), axis=1)

    # Create DocBins for exporting to Spacy format
    train_docbin = DocBin()
    test_docbin = DocBin()

    for doc in train_docs:
        train_docbin.add(doc)
    for doc in test_docs:
        test_docbin.add(doc)

    # Output training data in .spacy format
    train_corpus = f'{out_stem}_train.spacy'
    test_corpus = f'{out_stem}_test.spacy'

    train_docbin.to_disk(train_corpus)
    test_docbin.to_disk(test_corpus)



if __name__ == '__main__':
    import argparse
    input_files, output_stem = None, None
    nlp_model = None
    verbose = False

    parser = argparse.ArgumentParser(
        prog='prepare_training',
        description='Prepare training dataset from annotations.'
        )
    parser.add_argument("input_files", nargs='+', default=["data/el_session1.json", "data/el_session2.json"])
    parser.add_argument('-o', '--out', action="store_const", const=output_stem, help="Name structure of output files.",
                        default="data/el")
    parser.add_argument('-n', '--nlp', action="store_const", const=nlp_model, help="Name of the Spacy model",
                        default="en_core_web_lg")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args.input_files, args.out, args.verbose)