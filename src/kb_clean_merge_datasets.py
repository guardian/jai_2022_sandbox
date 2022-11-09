#!/usr/bin/env python
# coding: utf-8
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import re

# CONSTANTS
# TODO: rename to comply with convention
topics_translation_dict={
    'crime':'Crime',
    'crime.fraud':'Fraud',
    'crime.cyber':'Cybercrime',
    'crime.fin':'Financial crime',
    'crime.theft':'Theft',
    'crime.war':'War crimes',
    'crime.boss':'Criminal leadership',
    'crime.terror':'Terrorism',
    'crime.traffick':'Trafficking',
    'crime.traffick.drug':'Drug trafficking',
    'crime.traffick.human':'Human trafficking',
    'corp.offshore':'Offshore',
    'corp.shell':'Shell company',
    'gov':'Government',
    'gov.national':'National government',
    'gov.state':'State government',
    'gov.muni':'Municipal government',
    'gov.soe':'State-owned enterprise',
    'gov.igo':'Intergovernmental organization',
    'fin':'Financial services',
    'fin.bank':'Bank',
    'fin.fund':'Fund',
    'fin.adivsor':'Financial advisor',
    'role.pep':'Politician',
    'role.rca':'Close Associate',
    'role.judge':'Judge',
    'role.civil':'Civil servant',
    'role.diplo':'Diplomat',
    'role.lawyer':'Lawyer',
    'role.acct':'Accountant',
    'role.spy':'Spy',
    'role.oligarch':'Oligarch',
    'role.journo':'Journalist',
    'role.act':'Activist',
    'pol.party':'Political party',
    'pol.union':'Union',
    'rel':'Religion',
    'mil':'Military',
    'asset.frozen':'Frozen asset',
    'sanction':'Sanctioned entity',
    'debarment':'Debarred entity',
    'poi':'Person of interest'
}
more_cols_to_drop=[
                'bikCode',
                'dunsCode',
                'callSign',
                'tonnage',
                'grossRegisteredTonnage',
                'ogrnCode',
                'innCode',
                'leiCode',
                'swiftBic',
                'ogrnCode',
                'classification',
                'program',
                'sourceUrl',
                'addressEntity',
                'imoNumber',
                'mmsi',
                'registrationNumber',
                'modifiedAt',
                'idNumber',
                'passportNumber',
                'phone',
                'kppCode',
                'vatCode',
                'serialNumber',
                'owner',
                'opencorporatesUrl',
                'taxNumber',
                'flag',
                'status',
                'jurisdiction',
                'wikidataId',
                'email',
                'website',
                'education',
                'type',
                'firstName',
                'secondName',
                'createdAt',
                'middleName',
                'lastName',
                'title',
                'religion',
                'buildDate',
                'model',
                'incorporationDate',
                'previousName',
                'fatherName',
                'motherName',
                'address',
                'legalForm',
                ]
positions_in_full={'Min\.':'Minister',
                   'Dep\.':'Deputy',
                   'Pres\.':'President',
                   'Chmn\.':'Chairman',
                   'Dir\.':'Director',
                   'Cdr\.':'Commander',
                   'Sec\.':'Secretary',
                   'Gen\.':'General',
                   'Col\.':'Colonel',
                   'Brig\.':'Brigadier',
                   'Lt\.':'Lieutenant'}
cols_to_sentence={
    'gender':'This person is a ',
    'position':'This person has held these positions: ',
    'birthDate':'This person was born in ',
    'birthPlace': 'This person was born in ',
    'deathDate':'This person died in ',
    'keywords':'This person has worked in: ',
    'sector':'This person worked for: ',
    'publisher':'This person was present in ',
    'pastFlags':'In the past this person was at ',
    'ethnicity':'This person\'s ethnicity is '
}
context_cols=[
    'position',
    'gender',
    'birthdate',
    'country',
    'topics',
    'birthPlace',
    'nationality',
    'sector',
    'keywords',
    'deathdate',
    'publisher',
    'pastFlags',
    'ethnicity'
]
crime_vocab = ['murder',
               'fraud',
               'corruption',
               'conspiracy',
               'crime',
               'dealing',
               'drug',
               'trafficking',
               'criminal',
               'cheating',
               'forgery',
               'robbery',
               'violen',  # violent, violence
               'sexual',
               'rape',
               'assault',
               'illegal',
               'transport',
               'travel']
default_expr = 'is a '
crime_expr = 'was involved in '

# Utility functions
def url_generator(id_,name, dataset):
    """ Generate usable link for item in `dataset` from id (+name) """
    if dataset=='open_sanctions':
        return f'https://www.opensanctions.org/entities/{id_}'
    if dataset=='lilsis':
        return f'https://littlesis.org/person/{id_}-{name}'

def remove_columns(df, cols):
    """ Drop specific columns from data frame."""
    return df.drop(cols, axis=1)

def get_unique_properties(series):
    return list(set(series.explode().dropna().values))

def transform_into_sentence(df,col,sentence,separator=', '):
    """ Use `sentence` to create a sentence for each entry in `col` of `df`."""
    df.loc[~df[col].isna(),col]=df.loc[~df[col].isna(),col].apply(lambda x: f'{separator}'.join(x))
    df.loc[~df[col].isna(),col]=df.loc[~df[col].isna(),col].apply(lambda x: f'{sentence}{x}.')

def convert_country_code(row):
    """ convert each country code in `row` to the full country name"""
    import pycountry
    if not isinstance(row, list): return "" # Assume NAN then
    country_codes = filter(lambda c: len(c) == 2, row)
    countries = [pycountry.countries.get(alpha_2=country).name for country in country_codes
                 if pycountry.countries.get(alpha_2=country) is not None]
    return ','.join(c for c in countries)

def preprocess_open_sanctions(df, output_only=["PERSON"]):
    unique_keys = get_unique_properties(df["properties"])
    # Only keep entities with an entry on the name field
    df = df[df['properties'].apply(lambda x: "name" in x)]
    # df = df.drop(['properties','referents'],axis=1)

    # Standardise lower and upper cases in the kb names
    df.loc[:,'caption'] = df['caption'].str.title()
    properties_df = pd.DataFrame.from_dict(df['properties'].tolist())
    properties_df.replace('', np.nan, inplace=True)
    # Columns made up entirely of NaN
    cols_to_drop = properties_df.isna().all(0)
    cols_to_drop = list(cols_to_drop[cols_to_drop.values].index)
    properties_df.drop(cols_to_drop, axis=1, inplace=True)

    # Create copy of wikidataIDs
    wikidataIDs=properties_df['wikidataId'].copy().dropna()
    wikidataIDs=wikidataIDs.apply(lambda x: ', '.join(x))

    # Create copy of websites
    websites=properties_df['website'].copy().dropna()
    websites=websites.apply(lambda x: ', '.join(x))

    # Create sentences from topic memberships
    properties_df['topics'] = properties_df['topics'].fillna('').apply(
        lambda x: [f'Associated with {topics_translation_dict[key]}.' for key in x])
    properties_df['topics'] = properties_df['topics'].apply(lambda x: ' '.join(x))

    # Remove dates and ordinals from each string in the list [REFACTOR]
    date_expr = re.compile('\d{4}-\d{4}')
    digit_expr = re.compile('\s\d{2}[a-zA-Z]{2}\s')
    for expr in [date_expr, digit_expr]:
        # Delete expression from string
        properties_df.loc[~properties_df['position'].isna(), 'position'] = properties_df.loc[
            ~properties_df['position'].isna(), 'position'].apply(lambda x: [re.sub(expr, ' ', i) for i in x])
        # Remove parentheses and comma
        properties_df.loc[~properties_df['position'].isna(), 'position'] = properties_df.loc[
            ~properties_df['position'].isna(), 'position'].apply(
            lambda x: [i.replace('(', '').replace(')', '').replace(',', '') for i in x])
        # Remove position redundancy
        properties_df.loc[~properties_df['position'].isna(), 'position'] = properties_df.loc[
            ~properties_df['position'].isna(), 'position'].apply(lambda x: set(x))

    # Generate sentences for each pre-defined column of properties
    for col, sentence in cols_to_sentence.items():
        transform_into_sentence(properties_df, col, sentence)

    # Rename columns
    properties_df.rename(columns={"birthDate": "birthdate", 'deathDate':"deathdate"}, inplace=True)

    # Fix most common position abbreviations
    for abbv, full in positions_in_full.items():
        properties_df['position'] = properties_df['position'].str.replace(abbv, full, regex=False)

    # Convert country ISO alpha 2 codes into names [REFACTOR]
    properties_df['country'] = properties_df['country'].apply(convert_country_code)
    properties_df['nationality'] = properties_df['nationality'].apply(convert_country_code)

    # Transform country and nationality into sentences
    properties_df.loc[~properties_df['country'].isna(), 'country'] = properties_df.loc[
        ~properties_df['country'].isna(), 'country'].apply(lambda x: f'This person belongs to these countries: {x}.')
    properties_df.loc[~properties_df['nationality'].isna(), 'nationality'] = properties_df.loc[
        ~properties_df['nationality'].isna(), 'nationality'].apply(lambda x: f'This person has these nationalities: {x}.')

    # Create AKA column and drop AKA source columns
    properties_df['AKA'] = properties_df['name'] + properties_df['alias'] + properties_df['weakAlias']
    # properties_df.drop(['name', 'alias', 'weakAlias'], 1, inplace=True)

    # Generate single context text column from selected properties (see constants)
    properties_df['context'] = properties_df[context_cols].apply(
        lambda row: '. '.join(r for r in row.dropna()),
        axis=1
    )

    # Clean notes???
    properties_df['notes'] = properties_df['notes'].fillna('').apply(lambda x: ' '.join(x))
    # Adapt sentences for people involved in crime
    matches = properties_df['notes'].str.contains("(?:" + "|".join(fr"\b{cr}\b" for cr in crime_vocab) + ")",
                                        regex=True, flags=re.IGNORECASE)
    properties_df[matches]['notes'] = properties_df[matches]['notes'].str.lower().str.replace(default_expr, crime_expr)


    # Trim data frame in preparation for output
    properties_df = properties_df[['notes', 'AKA', 'context', 'birthdate', 'deathdate']]

    # Merge Wikipedia IDs and websites
    properties_df=properties_df.merge(wikidataIDs, how='left', left_index=True, right_index=True)
    properties_df=properties_df.merge(websites, how='left', left_index=True, right_index=True)

    df = df.merge(properties_df, how='left', left_index=True, right_index=True)
    df = df[df['schema'].str.upper().isin(output_only)]
    df = df.rename(columns={'caption': 'name'})
    df['full_notes'] = df['notes'].fillna('') + ' ' + df['context'].fillna('')

    # Remove obsolete columns
    df.drop(columns=['properties', 'referents'],
            inplace=True)

    return df

def preprocess_lilsis(df, output_only=["PERSON"]):

    # Get attributes for each entity and only work with entity types requested
    attributes_df = pd.DataFrame.from_dict(df['attributes'].tolist())
    attributes_df = attributes_df[attributes_df['primary_ext'].str.upper().isin(output_only)]

    # Generate description
    attributes_df['blurb'] = attributes_df['name'] + ' is a ' + attributes_df['blurb'] + '.'

    # Transform 'start date' as birth date in sentence
    attributes_df.loc[~attributes_df['start_date'].isnull(), 'start_date_sentence'] = attributes_df.loc[
        ~attributes_df['start_date'].isnull(), 'start_date'].apply(lambda x: f'This person was born in {x}.')

    # Transform 'end date' as death date into sentence
    attributes_df.loc[~attributes_df['end_date'].isnull(), 'end_date_sentence'] = attributes_df.loc[~attributes_df['end_date'].isnull(), 'end_date'].apply(
        lambda x: f'This person died in {x}.')

    # Transform types into sentence
    attributes_df['types'] = attributes_df['types'].apply(lambda x: ' '.join(x)).str.replace('Person', '')
    attributes_df['types'] = attributes_df['types'].str.replace(' ', ', ').apply(
        lambda x: 'This person is associated with: ' + x[1:] + '.')
    # Fix typos
    attributes_df['types'] = attributes_df['types'].str.replace(', ,', ',', regex=False).str.replace(', \.', '.', regex=False).str.replace('Media, ality',
                                                                                               'Media Personality', regex=False)
    # Remove sentences without context
    attributes_df.loc[attributes_df['types'] == 'This person is associated with: .', 'types'] = ''

    # Create context
    attributes_df['context'] = attributes_df['blurb'].fillna('') + ' ' + attributes_df['summary'].fillna('')
    attributes_df['context'] = attributes_df['context'].str.replace('\r', '').str.replace('\n', '')

    attributes_df.drop(['blurb', 'summary', 'updated_at', 'parent_id'], axis=1, inplace=True)

    attributes_df['extensions'] = attributes_df['extensions'].apply(lambda r: [r[key] for key in r.keys()])
    # unique_extension_keys_extension_keys = []
    # df = attributes_df['extensions'].apply(lambda r: list(combine_dictionaries(r).keys()))
    # for row in df:
    #     unique_extension_keys_extension_keys.extend(row)
    # del (df)

    # Add birthplace information and generate sentence
    valid_indices = attributes_df[attributes_df['extensions'].apply(lambda x: len(x)) > 0].index.values
    attributes_df.loc[valid_indices, 'birthplace'] = \
        attributes_df.loc[valid_indices, 'extensions'].apply(
        lambda x: [x[0]['birthplace'] if 'birthplace' in x[0] else None][0])

    attributes_df.loc[~attributes_df['birthplace'].isnull(), 'birthplace'] = \
        attributes_df.loc[~attributes_df['birthplace'].isnull(), 'birthplace'].apply(
        lambda x: f'This person was born in {x}.')

    # Add extra context
    extra_context_cols = ['start_date_sentence', 'end_date_sentence', 'types', 'birthplace']
    for col in extra_context_cols:
        attributes_df['context'] = attributes_df['context'] + ' ' + attributes_df[col].fillna('')

    # Remove obsolete columns
    attributes_df.drop(columns=["extensions"], inplace=True)

    # Preserve dataset information
    attributes_df['kb_origin'] = df['kb_origin']

    return attributes_df

def rename_columns(df, dataset):
    """ Standardise name of data frame."""
    desc_col = {'open_sanctions':'full_notes', 'lilsis':'context'}
    df = df.rename(columns={desc_col[dataset]: 'desc'})
    return df

def read_data(path, dataset_name_):
    """ Read in data and return dataframe"""
    if dataset_name_ == 'open_sanctions':
        df = pd.read_json(path, lines=True)
    elif dataset_name_ == 'lilsis':
        df = pd.read_json(path)
    else:
        raise NotImplementedError("Unknow dataset, cannot load.")
    logging.info(f"Read {df.shape[0]} lines and {df.shape[1]} columns.")
    return df

def main(input_files: List[Path], output_file: Path, verbose: bool = False):
    # Read in datasets
    inputs = []
    for inp in input_files:
        dataset_name = Path(inp).stem
        input_df = read_data(inp, dataset_name)
        input_df['kb_origin'] = dataset_name

        # Pre-process depending on dataset
        exclude_columns = {
            "open_sanctions": ['target', 'first_seen', 'last_seen', 'datasets'],
            "lilsis": []
        }
        input_df = remove_columns(input_df, exclude_columns[dataset_name])
        if dataset_name == 'open_sanctions':
            input_df = preprocess_open_sanctions(input_df)
            input_df = rename_columns(input_df, dataset_name)
        elif dataset_name == 'lilsis':
            input_df = preprocess_lilsis(input_df)
            input_df = rename_columns(input_df, dataset_name)
        else:
            raise NotImplementedError("Unknown dataset, no processing defined")

        # Add processed dataset to input list
        inputs.append(input_df)

    # Combine datasets into one
    kb_entities = pd.concat(inputs).reset_index().rename(columns={'index': 'original_index'})

    # Drop useless columns
    kb_entities.drop(['schema', 'notes', 'context', 'types', 'start_date_sentence', 'end_date_sentence'], axis=1,
                     inplace=True)

    # Fix trailing whitespaces
    kb_entities['desc'] = kb_entities['desc'].apply(lambda x: re.sub(r"\b(\.)[\.\s]+$", "\\1", x))

    # Remove entities with no description
    kb_entities = kb_entities[kb_entities['desc'].str.replace(' ', '').apply(len) > 0]

    # Drop duplicated based on same name and description
    kb_entities.drop_duplicates(subset=['name', 'desc'], inplace=True)

    # Find duplicate entries on the 'id' column and concatenate their descriptions
    duplicate_entities_by_id = kb_entities[kb_entities['id'].duplicated(keep=False)].sort_values(['id', 'name'])
    duplicate_entities_by_id = duplicate_entities_by_id[['id', 'desc']].groupby(
        ['id'])['desc'].apply(lambda x: ' '.join(x)).reset_index()

    # Remove those duplicates from dataset and replace them with concatenated description data
    kb_entities.drop(duplicate_entities_by_id.index, inplace=True)
    duplicate_entities_by_id = duplicate_entities_by_id[duplicate_entities_by_id['id'].duplicated(keep=False)]
    kb_entities = pd.concat([kb_entities, duplicate_entities_by_id])

    ###CODE BELOW NEEDS REFACTOR
    # Create auxiliary column for ordering based on len of desc field
    kb_entities['desc_len'] = kb_entities['desc'].str.len()

    redundancy_cols = ['birthdate', 'deathdate', 'website']
    for col in redundancy_cols:
        # Find duplicates ordered by description len
        redundant_entities_by_col = kb_entities[~(kb_entities[col].isna()) &
                                                (kb_entities.duplicated(['name', col], keep=False))
                                                ].sort_values(by=['name', 'desc_len'], ascending=False)

        # Drop duplicate_entities on name and birthdate
        redundant_entities_indices = redundant_entities_by_col.index
        kb_entities.drop(redundant_entities_indices, inplace=True)

        # Keep first of duplicated entities
        redundant_entities_by_col_consolidated_desc = redundant_entities_by_col.groupby(
            ['name', col]).first().reset_index()

        # Concatenate back to kb entity dataframe
        kb_entities = pd.concat([kb_entities, redundant_entities_by_col_consolidated_desc])
        kb_entities.reset_index(drop=True, inplace=True)


    # Ensure person name is included in all descriptions
    name_not_in_notes_indices = kb_entities[
        kb_entities.apply(lambda x: x['name'].lower() not in x['desc'].lower(), axis=1)].index.values
    naming_string = 'This person is called '
    kb_entities.loc[name_not_in_notes_indices, 'desc'] = kb_entities.loc[name_not_in_notes_indices].apply(
        lambda x: naming_string + x['name'] + '. ' + x['desc'], axis=1)

    # Ensure descriptions end in stop mark.
    kb_entities.loc[~kb_entities['desc'].isna(), 'desc'] = \
        kb_entities.loc[~kb_entities['desc'].isna(), 'desc'].apply(
        lambda x: x + '.' if x[-1] != '.' else x)

    # Clean up data by removing multiple trailing stop marks
    multi_stopmarks_expr = re.compile('\.\s?\.')

    for expr in [multi_stopmarks_expr]:
        # Replace expression in string
        kb_entities.loc[~kb_entities['desc'].isna(), 'desc'] = kb_entities.loc[
            ~kb_entities['desc'].isna(), 'desc'].apply(lambda x: ''.join([re.sub(expr, '. ', x)]))

    # Add KB urls
    kb_entities['kb_url'] = kb_entities[['id', 'name', 'kb_origin']].apply(
        lambda x: url_generator(*x), axis=1
    )

    # Output dataset
    kb_entities.to_csv(output_file)




if __name__ == '__main__':
    import argparse
    output_file, nlp_model = None, None
    verbose = False

    parser = argparse.ArgumentParser(
        prog='kb_clean_merge_datasets',
        description='Cleans and merges knowledge bases.'
        )
    parser.add_argument("input_files", nargs='+', default=["data/open_sanctions.json", "data/lilsis.json"])
    parser.add_argument('-o', '--out', action="store_const", const=output_file, help="Name of output file.",
                        default="data/kb_entities_full.csv")
    parser.add_argument('-n', '--nlp', action="store_const", const=nlp_model, help="Name of the Spacy model",
                        default="en_core_web_lg")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args.input_files, args.out, args.verbose)