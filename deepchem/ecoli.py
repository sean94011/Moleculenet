import os
import logging
import deepchem as dc
from deepchem.models.graph_models import GraphConvModel
from deepchem.utils import download_url
import sklearn.metrics

logger = logging.getLogger(__name__)
ECOLI_URL = 'https://raw.githubusercontent.com/yangkevin2/coronavirus_data/master/data/ecoli.csv'

def load_ecoli(featurizer='GraphConv',
               split='random',
               reload=True,
               K=4,
               data_dir=None,
               save_dir=None,
               **kwargs):

    
    download_url("https://raw.githubusercontent.com/yangkevin2/coronavirus_data/master/data/ecoli.csv")
    data_dir = os.path.join(dc.utils.get_data_dir())
    dataset_file= os.path.join(dc.utils.get_data_dir(), "ecoli.csv")

    featurizer = dc.feat.ConvMolFeaturizer()
    ecoli_tasks = ['activity']

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = dc.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = dc.feat.AdjacencyFingerprint(
            max_n_atoms=150, max_valence=6)
    elif featurizer == "smiles2img":
        img_size = kwargs.get("img_size", 80)
        img_spec = kwargs.get("img_spec", "std")
        featurizer = dc.feat.SmilesToImage(
            img_size=img_size, img_spec=img_spec)

    loader = dc.data.CSVLoader(tasks = ecoli_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    splitters = {
        'index': dc.splits.IndexSplitter(),
        'random': dc.splits.RandomSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter(),
        'butina': dc.splits.ButinaSplitter(),
        'task': dc.splits.TaskSplitter(),
        'stratified': dc.splits.RandomStratifiedSplitter()
    }
    splitter = splitters[split]

    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)
    all_dataset = (train, valid, test)

    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=train)
    ]

    logger.info("About to transform data")
    for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

    return ecoli_tasks, all_dataset, transformers

ecoli_tasks, ecoli_datasets, transformers = load_ecoli(featurizer='GraphConv', reload=False)
train_dataset, valid_dataset, test_dataset = ecoli_datasets


n_tasks = len(ecoli_tasks)
model = GraphConvModel(n_tasks, batch_size=50, mode='classification')

num_epochs = 30
losses = []
import numpy as np
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

for i in range(num_epochs):
    loss = model.fit(train_dataset, nb_epoch=1)
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)
    print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])
    test_scores = model.evaluate(test_dataset, [metric], transformers)
    print("Test ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])
    print(" ")

test_predictions = model.predict(test_dataset)
for i in range(1):
    tp, fp, threshold = metrics.roc_curve(test_dataset.y[:,i], test_predictions[:,i,0])
    print('TPR: ', tp, '\nFPR: ', fp, '\n')    