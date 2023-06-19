import os
import lightgbm as lgb
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def get_data(path, debug=False, conf=None, test_num=None):
    target = conf['target']
    num_cols = list(map(lambda x: x.split(':')[0], conf['using_features']['numerical']))

    def get_schema():
        ret = {'session_id': str, target: int}
        return ret

    df = pd.read_csv(path,
                     converters=get_schema(),
                     nrows=1000 if debug else test_num,
                     usecols=None if test_num else ['session_id', target] + num_cols)
    print(f'loading pandas dataFrame from {path}, num of rows: {len(df)}.')
    return df


def read_conf(conf_path):
    with open(conf_path, 'r') as f:
        d = json.load(f)
        return d


def evaluate(labels, preds):
    max_f1 = float('-inf')
    labels, preds = np.array(labels), np.array(preds)
    for th in np.linspace(0, 1, 50):
        x = (preds > th).astype('int')
        f1 = metrics.f1_score(labels, x)
        max_f1 = max(f1, max_f1)

    precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
    pre_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    idx = [np.argmin(np.abs(precision[:-1] - p)) for p in pre_thresh]
    df = pd.DataFrame({'precision': precision[idx], 'recall': recall[idx], 'thresholds': thresholds[idx]})
    df = pd.DataFrame(df.values.T, index=df.columns, columns=pre_thresh)

    return {'auc': metrics.roc_auc_score(labels, preds),
            'f1-score': max_f1,
            'acc': metrics.accuracy_score(labels, preds > 0.5),
            'report': '\n{}'.format(metrics.classification_report(labels, preds > 0.5)),
            'confusion': '\n{}'.format(metrics.confusion_matrix(labels, preds > 0.5)),
            'pr-curve': '\n{}'.format(df)
            }


def plot_pred_dist(labels, preds, save_path=None):
    path, name = os.path.split(save_path)

    # distribution
    x = pd.DataFrame({"labels": labels, "preds": preds})
    class_zero_preds = x.loc[x["labels"] == 0, "preds"]
    class_one_preds = x.loc[x["labels"] == 1, "preds"]

    plt.figure()
    plt.hist(x['preds'], bins=20, label='all', alpha=0.2)
    plt.hist(class_zero_preds, bins=20, label="0", alpha=0.5)
    plt.hist(class_one_preds, bins=20, label="1", alpha=0.5)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(path, 'dist_{}'.format(name)))

    # pr-curve
    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    if save_path:
        plt.savefig(os.path.join(path, 'pr_curve_{}'.format(name)))
    plt.close()


def feature_importance(model, feature_cols):
    f_imp = zip(feature_cols, model.feature_importance(importance_type='gain'))
    df = pd.DataFrame(f_imp, columns=['feature', 'score'])
    return df.sort_values(by='score', ascending=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", nargs=2, help="path to training and dev dataset")
    parser.add_argument("--config", "-f", help="path to training config")
    parser.add_argument("--model_dir", "-m", help="path to the model dir")
    parser.add_argument("--predict", "-p", default=None, help="path to test dataset")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    CONF = read_conf(args.config)

    print('loading data...')
    train_df, dev_df = get_data(args.data[0], conf=CONF), get_data(args.data[1], conf=CONF)
    num_cols = list(map(lambda x: x.split(':')[0].strip(), CONF['using_features']['numerical']))
    target = CONF['target']

    train_data = lgb.Dataset(train_df[num_cols],
                             label=train_df[target],
                             feature_name=num_cols)

    model = lgb.train(CONF['training_params'], train_set=train_data)
    preds = model.predict(dev_df[num_cols])

    # evaluate gbdt model
    eval_metrics = evaluate(dev_df[target], preds)
    with open(os.path.join(args.model_dir, "metrics.json"), 'w') as f:
        json.dump(eval_metrics, f)

    print('=' * 20)
    print("Eval gbdt results")
    for key in sorted(eval_metrics.keys()):
        print("  {} = {}".format(key, str(eval_metrics[key])))

    plot_pred_dist(dev_df[target], preds, save_path=os.path.join(args.model_dir, 'eval.png'))

    ret_df = feature_importance(model, num_cols)
    ret_df.to_csv(os.path.join(args.model_dir, "feature_importance.csv"), index=False)
    model.save_model(filename=os.path.join(args.model_dir, "lgb.model.txt"))
    with open(os.path.join(args.model_dir, 'train.conf.json'), 'w') as f:
        json.dump(CONF, f)

    dev_df['preds'] = preds
    dev_df.to_csv(os.path.join(args.model_dir, "dev_preds.csv"), index=False)

    if args.predict is not None:
        print('=' * 20)
        print("Test gbdt results")
        test_df = get_data(args.predict, conf=CONF, test_num=1000000)
        preds_test = model.predict(test_df[num_cols])
        test_metrics = evaluate(test_df[target], preds_test)
        with open(os.path.join(args.model_dir, "metrics_test.json"), 'w') as f:
            json.dump(test_metrics, f)

        for key in sorted(test_metrics.keys()):
            print("  {} = {}".format(key, str(test_metrics[key])))

        plot_pred_dist(test_df[target], preds_test, save_path=os.path.join(args.model_dir, 'test.png'))

        test_df['preds'] = preds_test
        test_df.to_csv(os.path.join(args.model_dir, "test_preds.csv"), index=False)


if __name__ == '__main__':
    main()
