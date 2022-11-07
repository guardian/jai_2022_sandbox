#!/usr/bin/env python
# coding: utf-8
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


#
# ## Pre-process individual kb dataset
# # Select the intended kb dataset
# #dataset='lilsis'
# dataset='open_sanctions'# Read csv file
# kb_entities=pd.read_csv(f'{dataset}_entities.csv',index_col=0)# Select only KB entries with a person entity
# person_named_entities_name='Person'
# person_named_entities_col_d={'open_sanctions':'schema', 'lilsis':'primary_ext'}
# kb_entities=kb_entities[
#     kb_entities[
#         person_named_entities_col_d[dataset]]==person_named_entities_name
# ]# Change column names
# desc_col_d={'open_sanctions':'full_notes', 'lilsis':'context'}
# desc_col=desc_col_d[dataset]
# kb_entities=kb_entities.rename(columns={desc_col:'desc'})# Drop entities without a description
# kb_entities.dropna(subset=['desc'],inplace=True)# Reorder columns
# if dataset!='open_sanctions':
#     kb_entities=kb_entities.rename(columns={'aliases':'AKA'})
# kb_entities=kb_entities[['id','name','desc', 'AKA']]# Remove cyrillic
# cyrillic = "вгдеёзийклмнопрстуфхъыьэАБВГДЕЁЗИЙКЛМНОПРСТУФХЪЫЬЭ"
# for i,symbol in enumerate(cyrillic):
#     cyrillic_condition=(kb_entities['name'].str.contains(symbol))
#     if i==0:
#         cyrillic_df=kb_entities[cyrillic_condition]
#     else:
#         cyrillic_df=cyrillic_df.append(kb_entities[cyrillic_condition])
#         cyrillic_df=cyrillic_df.drop_duplicates()
# kb_entities=kb_entities.drop(cyrillic_df.index)kb_entities.to_csv(f'../kb_datasets/kb_entities_{dataset}.csv')
#
# # Read both datasets
# os_kb_entities=pd.read_csv(f'kb_entities_open_sanctions.csv',index_col=0)
# os_kb_entities['kb_origin']='open_sanctions'
# ls_kb_entities=pd.read_csv(f'kb_entities_lilsis.csv',index_col=0)
# ls_kb_entities['kb_origin']='lilsis'
#
#
# # In[17]:
#
#
# # Combine datasets into one
# kb_entities=pd.concat([os_kb_entities,ls_kb_entities]).reset_index().rename(columns={'index':'original_index'})
#
#
# # In[18]:
#
#
# kb_entities.shape
#
#
# # In[19]:
#
#
# kb_entities.head(2)
#
#
# # In[20]:
#
#
# ## Resolve duplicate entitiy IDs
#
#
# # In[21]:
#
#
# # Find duplicate entries on the 'id' columns
# redudant_entities_by_id=kb_entities[kb_entities['id'].duplicated(keep=False)].sort_values(['id','name'])
#
#
# # In[22]:
#
#
# # Drop duplicate_entities (these were all cases where the ID was taken from Wikidata)
# redudant_entities_indices=redudant_entities_by_id.index
# kb_entities.drop(redudant_entities_indices, inplace=True)
#
#
# # In[23]:
#
#
# # Combine desc across duplicated entities
# redudant_entities_by_id_consolidated_desc=redudant_entities_by_id[['id','desc']].groupby(['id'])['desc'].apply(lambda x: ' '.join(x)).reset_index()
#
#
# # In[24]:
#
#
# # Drop all duplicated apart from the first (needs to be sorted by [id, name])
# redudant_entities_by_id.drop_duplicates(subset=['id'],keep='first',inplace=True)
#
#
# # In[25]:
#
#
# # Concatenate back to kb entity dataframe
# kb_entities=pd.concat([kb_entities,redudant_entities_by_id])
#
#
# # In[26]:
#
#
# # Cast names in lower case
# kb_entities['name']=kb_entities['name'].str.lower()
#
#
# # In[27]:
#
#
# # Drop duplicates on name and desc
# kb_entities.drop_duplicates(['name','desc'],inplace=True)
#
#
# # In[28]:
#
#
# ## Clean up data
#
# # Remove titles
# kb_entities['name']=kb_entities['name'].str.split('\. ').apply(lambda x: ' '.join([s for s in x if len(s)>3]))kb_entities['first_name']=kb_entities['name'].apply(lambda r: r.split(' ')[0])kb_entities['last_name']=kb_entities['name'].apply(lambda r: r.split(' ')[-1])kb_entities[kb_entities.duplicated(subset=['first_name','last_name','desc'], keep=False)].sort_values(by=['first_name','last_name']).to_csv('duplicate_entities_on_first_and_last_names.csv')
# # In[29]:
#
#
# # Remove special characters (cyrillic, chinese, japanese, arabic, numbers)
#
# # Remove cyrillic from descriptions
# cyrillic = "вгдеёзийклмнопрстуфхъыьэАБВГДЕЁЗИЙКЛМНОПРСТУФХЪЫЬЭжцчщюяєії"
# for symbol in cyrillic:
#     kb_entities['desc']=kb_entities['desc'].str.replace(symbol,'')# Get all special characters from description column
# special_characters=set()
# for row in kb_entities.iterrows():
#     special_characters.update(
#         set(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is None, row[1][3])
#            )
#     )# Get set of all unique words in description column
# vocab=set()
# for row in kb_entities.iterrows():
#     vocab.update(
#         set(row[1][3].split(' ')
#            )
#     )clean_vocab=set()
# for word in vocab:
#     clean_word=word
#     for character in special_characters:
#         clean_word=clean_word.replace(character,'')
#     clean_vocab.add(clean_word)pd.DataFrame(clean_vocab,columns=['vocab']).to_csv('clean_vocab.csv')
# # In[30]:
#
#
# clean_vocab=pd.read_csv('clean_vocab.csv',index_col=0)
# clean_vocab=set(clean_vocab['vocab'])
#
#
# # In[31]:
#
#
# kb_entities['desc']=kb_entities['desc'].apply(lambda x: ' '.join([word for word in x.split(' ') if word in clean_vocab]))
#
#
# # In[32]:
#
#
# kb_entities
#
#
# # In[33]:
#
#
# ## Resolve non-id ambiguities
#
# kb_entities.loc[kb_entities['name'].duplicated(keep=False),'name'].value_counts()import spacy
# nlp = spacy.load("en_core_web_lg")from numpy import dot
# from numpy.linalg import norm
# def calculate_cosine_similarity(descriptions_vec,vector_ref_sentence):
#     """
#     Return a dictionary mapping the kb entity id to cosine similarity score
#     between kb embedded descriptions and the reference vector.
#     """
#
#     score=np.nan_to_num(
#         dot(vector_ref_sentence, descriptions_vec)/
#         (norm(vector_ref_sentence)*norm(descriptions_vec))
#     ,0)
#     return scorekb_entities['desc_enc']=kb_entities['desc'].apply(lambda x: nlp(x).vector)kb_entities.to_csv('kb_entities_full.csv')kb_entities[kb_entities.duplicated(subset=['last_name'], keep=False)].sort_values(by=['last_name','name']).to_csv('duplicates_by_last_name.csv')for i,name in enumerate(kb_entities['name'].unique()):
#     df=kb_entities[kb_entities['name']==name]
#     df=df[['name','id','desc','desc_enc']]
#     df=df.merge(df, on='name', suffixes=['_1','_2'])
#     df['similarity_score']=df.apply(
#         lambda x: calculate_cosine_similarity(x['desc_enc_1'],x['desc_enc_2']),
#             1
#         )
#     df=df[df['similarity_score']>0]
#     if i==0:
#         similarity_df=df[['id_1','id_2','similarity_score']]
#     else:
#         similarity_df=pd.concat([similarity_df, df[['id_1','id_2','similarity_score']]])
#
# similarity_df=similarity_df[(similarity_df['id_1']!=similarity_df['id_2'])]# The merging the ids agains themselves creates deplicates based on order
# # (id 1, id 2 or id 2, id 1 doesn't matter, just the pairing of the set)
# # This code removes redudant rows
# similarity_df['id_1']=similarity_df['id_1'].astype(str)
# similarity_df['id_2']=similarity_df['id_2'].astype(str)
# similarity_df['id_pair']=similarity_df.apply(lambda x: ' '.join(set([x['id_1'],x['id_2']])),1)
# similarity_df=similarity_df.drop_duplicates('id_pair',keep='first')# Fetch the context for each id
# similarity_df=similarity_df.merge(kb_entities[['name','id','desc']], how='left',left_on='id_1', right_on='id')
# similarity_df.drop('id',1,inplace=True)
# similarity_df=similarity_df.rename(columns={'name':'name_1','desc':'desc_1'})
# similarity_df=similarity_df.merge(kb_entities[['name','id','desc']], how='left',left_on='id_2', right_on='id')
# similarity_df.drop('id',1,inplace=True)
# similarity_df=similarity_df.rename(columns={'name':'name_2','desc':'desc_2'})similarity_df.to_csv('similarity_df.csv')
# # In[34]:
#
#
# ## Add KB URLs
#
#
# # In[35]:
#
#
#
#
#
# # In[36]:
#
#
# kb_entities['kb_url']=kb_entities.apply(
#     lambda x: [url_generator(x['id'],x['name'], 'open_sanctions')
#                              if x['kb_origin']=='open_sanctions'
#                              else url_generator(x['id'],x['name'], 'lilsis')
#                                                   ][0],1
# )
#
#
# # In[43]:
#
#
# kb_entities[kb_entities['desc'].isna()]
#
#
# # In[40]:
#
#
# kb_entities[kb_entities.duplicated(keep=False,subset=['name'])].sort_values(by=['name']).to_csv(f'duplicate_full_name_aliases.csv')
#
#
# # In[44]:
#
#
# kb_entities.to_csv('kb_entities_full.csv')
################################################################
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

    # properties_df.drop(more_cols_to_drop, 1, inplace=True)

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
        properties_df['position'] = properties_df['position'].str.replace(abbv, full)

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
    matches = properties_df['notes'].str.contains("(" + "|".join(fr"\b{cr}\b" for cr in crime_vocab) + ")",
                                        regex=True, flags=re.IGNORECASE)
    properties_df['notes'] = properties_df[matches]['notes'].str.lower().str.replace(default_expr, crime_expr)


    # Trim data frame in preparation for output
    properties_df = properties_df[['notes', 'AKA', 'context', 'birthdate', 'deathdate']]

    ## Dropped the nans, INDEX MISALIGNED!
    properties_df=properties_df.merge(wikidataIDs,left_index=True, right_index=True)
    properties_df=properties_df.merge(websites,left_index=True, right_index=True)

    df = df.merge(properties_df, left_index=True, right_index=True)
    df = df[df['schema'].str.upper().isin(output_only)]
    df = df.rename(columns={'caption': 'name'})
    df['full_notes'] = df['notes'].fillna('') + ' ' + df['context'].fillna('')

    return df



def main(input_files: List[Path], output_file: Path, verbose: bool = False):
    # Read in datasets
    inputs = []
    for inp in input_files:
        tmp = pd.read_json(inp, lines=True)
        dataset_name = Path(inp).stem
        tmp['kb_origin'] = dataset_name

        # Pre-process depending on dataset
        exclude_columns = {
            "open_sanctions": ['target', 'first_seen', 'last_seen', 'datasets'],
            "lilsis": []
        }
        tmp = remove_columns(tmp, exclude_columns[dataset_name])
        if dataset_name == 'open_sanctions':
            tmp = preprocess_open_sanctions(tmp)
        elif dataset_name == 'lilsis':
            continue
        else:
            raise NotImplementedError("Unknown dataset, no processing defined")

        # Add processed dataset to input list
        inputs.append(tmp)

    # Combine datasets into one
    kb_entities = pd.concat(inputs).reset_index().rename(columns={'index': 'original_index'})

    # Output dataset
    kb_entities.to_csv(output_file)

if __name__ == '__main__':
    import argparse
    input_files, output_file = None, None
    verbose = False

    parser = argparse.ArgumentParser(
        prog='kb_clean_merge_datasets',
        description='Cleans and merges knowledge bases.'
        )
    parser.add_argument("input_files", nargs='+', default=["data/open_sanctions.json", "data/lilsis.json"])
    parser.add_argument('-o', '--out', action="store_const", const=output_file, help="Name of output file.",
                        default="kb_entities_full.csv")  # option that takes a value
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args.input_files, args.out, args.verbose)