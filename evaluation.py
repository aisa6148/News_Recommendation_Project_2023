import MMR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ndcg_score

def _diversity_user(glove, news_df, recs):
    score = 0.0
    count = 0.0
    for i in range(len(recs)):
        for j in range(i+1, len(recs)):
            count += 1.0
            score += MMR.get_category_similarity(glove, news_df, recs[i], recs[j])
    return score/count
        
# df must have a columns named "news_id" containing a list of news_ids
def diversity_eval(glove, news_df, df):
    diversity_score = 0.0
    count = 0.0
    for index, row in df.iterrows():
        diversity_score += _diversity_user(glove, news_df, row['news_id'])
        count += 1.0
    return diversity_score/count

# data frame must have columns "label" and "pred" which each
# contain a list of the same size of size at least k
def calculate_ndcg_at_k(df, k):
    avg_ndcg = 0
    count = 0
    for i in range(len(df)):
        temp = df.iloc[i]
        ndcg = ndcg_score([temp["label"]], [temp["pred"]], k=k)
        avg_ndcg += ndcg
        count += 1
    return avg_ndcg/count

# data frame must have columns "label" and "pred" which each
# contain a list of the same size of size
def calculate_auc(df):
    return roc_auc_score(df["label"], df["pred"])

def graph_ndcg(ndcgs, lamdas, k, model_name):
    plt.plot(lamdas, ndcgs)
    plt.title(f"NDCG@{k} based on lambda: {model_name}")
    plt.xlabel("Lambda")
    plt.ylabel(f"NDCG@{k}")
    plt.show()

def graph_diversity(diversities, lamdas, model_name):
    plt.plot(lamdas, diversities)
    plt.title(f"Diversity based on lambda: {model_name}")
    plt.xlabel("Lambda")
    plt.ylabel("Diversity")
    plt.show()