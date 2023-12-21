import os
import tempfile
import torch
import torchtext
import urllib
import zipfile

import pandas as pd
import numpy as np

# The dataset is split into training and validation set, each with a large and small version.
# The format of the four files are the same.
# For demonstration purpose, we will use small version validation set only.
base_url = 'https://mind201910small.blob.core.windows.net/release'
training_small_url = f'{base_url}/MINDsmall_train.zip'
validation_small_url = f'{base_url}/MINDsmall_dev.zip'
training_large_url = f'{base_url}/MINDlarge_train.zip'
validation_large_url = f'{base_url}/MINDlarge_dev.zip'

def _download_url(url,
                 temp_dir,
                 destination_filename=None,
                 progress_updater=None,
                 force_download=False,
                 verbose=True):
    """
    Download a URL to a temporary file
    """
    if not verbose:
        progress_updater = None
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(
                os.path.basename(url)))
        return destination_filename
    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(url),
                                                 destination_filename),
              end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose:
        print('...done, {} bytes.'.format(nBytes))
    return destination_filename

def get_news_df():
    temp_dir = os.path.join(tempfile.gettempdir(), 'mind')
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = _download_url(validation_small_url, temp_dir, verbose=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    os.listdir(temp_dir)

    news_path = os.path.join(temp_dir, 'news.tsv')
    news_df = pd.read_table(news_path,
                header=None,
                names=[
                    'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                    'title_entities', 'abstract_entities'
                ])
    news_df.set_index("id", inplace=True)
    return news_df

# only needs to be called once
def load_glove():
    return torchtext.vocab.GloVe(name="6B", dim=50)

# nid_1 and nid_2 are news ids (strings)
# returns similarity of categories of nid_1 and nid_2 using cosine similarity
# if categories don't exist returns 0
def get_category_similarity(glove, news_df, nid_1, nid_2):
    if nid_1 not in news_df.index or nid_2 not in news_df.index:
        return 0
    
    cat1 = news_df.loc[nid_1]["category"]
    cat2 = news_df.loc[nid_2]["category"]
    
    return 1 - torch.cosine_similarity(glove[cat1].unsqueeze(0), glove[cat2].unsqueeze(0)).item()

# Calculates mmr score for a given item
# item: news id
# pred: relevance of item
# recs_so_far: list of news ids recommended so far
def _mmr_item(glove, news_df, item, pred, recs_so_far, lamda):
    return (lamda * pred) - (1 - lamda) * np.max([1 - get_category_similarity(glove, news_df, item, x) for x in recs_so_far])

# Calculates list of recommendations
# recs is a list of news ids
# pred scores is a list of relevance scores, same order as recs
# lamda is a weight parameter
# k is how many items should be in the recommendation; assume k >= 1
def _mmr_user(glove, news_df, recs, pred_scores, lamda, k):
    list_so_far = [recs[0]]
    preds_so_far = [pred_scores[0]]
    while len(list_so_far) < k:
        max_mmr = -2 # mmr can range from -1 to 1
        max_mmr_id = ''
        for i in range(0, len(recs)): #should be a better way to do this
            if recs[i] not in list_so_far:
                mmr_score = _mmr_item(glove, news_df, recs[i], pred_scores[i], list_so_far, lamda)
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    max_mmr_id = recs[i]
        list_so_far.append(max_mmr_id)
        preds_so_far.append(max_mmr)
    return list_so_far, preds_so_far
        
# Calculates recommendations according to mmr for all users
# df is a Pandas dataframe with cols user, news_id, pred where pred[i] is the relevance score for news_id[i]
# lamda is a weight parameter
# k is how many items should be in the recommendation; assume k >= 1
def mmr_all(glove, news_df, df, lamda, k):
    result_df = {}
    for index, row in df.iterrows():
        news_ids, preds = _mmr_user(glove, news_df, row['news_id'], row['pred'], lamda, k)
        result_df[index] = {"news_id": news_ids, "pred": preds}
    return result_df