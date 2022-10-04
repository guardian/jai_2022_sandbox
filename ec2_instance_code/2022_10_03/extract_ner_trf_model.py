import glob
import logging
import pandas as pd
import spacy
logging.basicConfig(level=logging.DEBUG)
spacy.prefer_gpu()

def get_data(doc_index,doc,ent_types):
    """
    Extract the entity data (text, label, start, end, start_char, end_char) 
    from a Spacy Doc and format into JSON.
    Filter output to only include `ent_types`.
    :returns dict
    """
    ents = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start,
            "end": ent.end,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in ent_types
    ]
    return {"doc_index":doc_index,
            #"text": doc.text, 
            "ents": ents}

logging.info('Starting NER extraction')

logging.info('Loading Spacy model')
NER_TRF_MODEL="en_core_web_trf"
nlp = spacy.load(NER_TRF_MODEL)
ent_types = nlp.pipe_labels["ner"]
unwanted_ent_types=['CARDINAL','LANGUAGE','ORDINAL','PERCENT','QUANTITY','TIME']
ent_types = [ent for ent in ent_types if ent not in unwanted_ent_types]
csv_file_list=glob.glob('/home/ubuntu/JAI/data/*.csv')
csv_file_list.sort()
#start on most recent year
csv_file_list.reverse()

logging.info('Starting iteration through csv files')
for csv_file in csv_file_list:
    csv_file_name=''.join(csv_file.split('/')[-1].split('.')[-2])
    export_csv_file=f'/home/ubuntu/JAI/data/extracted_named_entities_output/{csv_file_name}_ner.csv.gz'
    if glob.glob(export_csv_file):
        # Stop entity extraction for files already processed
        continue
    logging.info('------------------------')
    logging.info(f'Reading {csv_file} data')
    try:
        data=pd.read_csv(csv_file)
    except:
        continue
    #data=data.iloc[:100]
    data['body_text']=data['body_text'].astype('str')
    # Ensure incremental ordered index to reference back to articles in the dataset
    data=data.reset_index(drop=True).sort_index()
    data=data.to_dict('index')
    gu_article_list=[data[key]['body_text'] for key in data.keys()]
    response_body = []
    exceptions=[]
    for doc_index,doc in enumerate(nlp.pipe(gu_article_list, batch_size=20)):
        if doc_index%1000==0:
            logging.info(f'Extracting named entities from {csv_file} article {doc_index}')
        response_body.append(get_data(doc_index,doc, ent_types))
    d={}
    i=-1
    for response in response_body:
        for ent_ind,ent in enumerate(response['ents']):
            i+=1
            ent['doc_index']=response['doc_index']
            d[i]=ent
    df=pd.DataFrame.from_dict(d,orient='index')
    df.to_csv(export_csv_file)
    logging.info(f'Finished processing {csv_file}')
    logging.info('------------------------')
