import numpy as np
import pandas as pd
from sklearn import metrics


TRUE_OVER_TIME = 10
FALSE_OVER_TIME = 90


def sample_anlysis(df, length):
    dfl = df[(df['label'] == 1) & (df['preds'] < 0.1)]
    print('[left] len: {} head: \n{}'.format(len(dfl), dfl.head(length)))
    dfr = df[(df['label'] == 0) & (df['preds'] > 0.9)]
    print('[right] len: {} head: \n{}'.format(len(dfr), dfr.head(length)))


def dist_anlysis(test, mode='test'):
    test_svr_wait_true_len = len(test[(test['label'] == 1) & (test['servicer_wait_time'] > 60)])
    test_svr_wait_false_len = len(test[(test['label'] == 0) & (test['servicer_wait_time'] > 60)])
    print('[svr_wait>60] true: {}\tfalse: {}\tmultiple: {:.4f}'.format(
        test_svr_wait_true_len, test_svr_wait_false_len, test_svr_wait_true_len / test_svr_wait_false_len))

    test_usr_wait_true_len = len(test[(test['label'] == 1) & (test['user_wait_time'] > 10)])
    test_usr_wait_false_len = len(test[(test['label'] == 0) & (test['user_wait_time'] > 10)])
    print('[usr_wait>10] true: {}\tfalse: {}\tmultiple: {:.4f}'.format(
        test_usr_wait_true_len, test_usr_wait_false_len, test_usr_wait_true_len / test_usr_wait_false_len))

    test_sr_svr_true_len = len(test[(test['label'] == 1) & (test['send_role'] == 3)])
    test_sr_svr_false_len = len(test[(test['label'] == 0) & (test['send_role'] == 3)])
    print('[send_role=3] true: {}\tfalse: {}\tmultiple: {:.4f}'.format(
        test_sr_svr_true_len, test_sr_svr_false_len, test_sr_svr_true_len / test_sr_svr_false_len))

    test_sr_usr_true_len = len(test[(test['label'] == 1) & (test['send_role'] == 1)])
    test_sr_usr_false_len = len(test[(test['label'] == 0) & (test['send_role'] == 1)])
    print('[send_role=1] true: {}\tfalse: {}\tmultiple: {:.4f}'.format(
        test_sr_usr_true_len, test_sr_usr_false_len, test_sr_usr_true_len / test_sr_usr_false_len))

    test_pol_true_len = len(test[(test['label'] == 1) & (test['polling'] > 400)])
    test_pol_false_len = len(test[(test['label'] == 0) & (test['polling'] > 400)])
    print('[polling>400] true: {}\tfalse: {}\tmultiple: {:.4f}'.format(
        test_pol_true_len, test_pol_false_len, test_pol_true_len / test_pol_false_len))

    if mode == 'test':
        test_none_true = len(test[(test['label'] == 1) & (test['preds'] < 0.1) & (test['servicer_message_time'] < TRUE_OVER_TIME) & (test['user_message_time'] < TRUE_OVER_TIME)])
        test_none_false = len(test[(test['label'] == 0) & (test['preds'] > 0.9) & (test['servicer_message_time'] > FALSE_OVER_TIME) & (test['user_message_time'] > FALSE_OVER_TIME)])
        print('[svr|usr<10] true: {}'.format(test_none_true))
        print('[svr|usr>60] false: {}'.format(test_none_false))


def compute_metrics(labels, scores, thresh, show=False):
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    pre_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    idx = [np.argmin(np.abs(precision[:-1] - p)) for p in pre_thresh]
    df = pd.DataFrame({'precision': precision[idx], 'recall': recall[idx], 'thresholds': thresholds[idx]})
    df = pd.DataFrame(df.values.T, index=df.columns, columns=pre_thresh)
    res = {
        'acc': metrics.accuracy_score(labels, scores > thresh),
        'auc': metrics.roc_auc_score(labels, scores > thresh),
        'f1-score': metrics.f1_score(labels, scores > thresh),
        'report': '\n{}'.format(metrics.classification_report(labels, scores > thresh)),
        'confusion': '\n{}'.format(metrics.confusion_matrix(labels, scores > thresh)),
        'pr-curve': '\n{}'.format(df),
    }
    if show:
        for key in sorted(res.keys()):
            print("{} = {}".format(key, str(res[key])))
    return res


def split_bridge_dataset(df, rt):
    return df.sample(frac=1.0).iloc[:int(len(df)*rt)]


if __name__ == '__main__':
    # 训练样本分布分析
    print('------- train样本分布分析 -------')
    test = pd.read_csv('dataset/smart_boost/train.csv')
#     test = pd.read_csv('dataset/smart_boost/d180_150w_20s_wash_label/train.csv')
    dist_anlysis(test, mode='train')

    # 测试样本分布分析
    print('\n------- test样本分布分析 -------')
    # test = pd.read_csv('ml_model/test_preds.csv')
    test = pd.read_csv('outputs/smart_boost_output/albert/checkpoint-37500/preds_dev.csv')
#     test = pd.read_csv('outputs/smart_boost_output/albert_small_256_150w/checkpoint-375000/preds_dev.csv')
    dist_anlysis(test)

    # test错误sample
    print('\n------- test错误sample -------')
    sample_anlysis(test, length=20)

    # 模型预测错误原因分析
    print('\n------- predict错误原因分析 -------')
    score_thresh = 0.5
    test_err = test[(((test['label'] == 0) & (test['preds'] > score_thresh)) | ((test['label'] == 1) & (test['preds'] < score_thresh)))]
    print('错误量: {} 错误率: {:.4f}'.format(len(test_err), len(test_err)/len(test)))
    dist_anlysis(test_err)

    # 模型效果
    print('\n------- 模型效果 -------')
    compute_metrics(test['label'], test['preds'], thresh=score_thresh, show=True)

    # 模型效果-修正label
    print('\n------- 模型效果-修正label -------')
    test.loc[(test['label'] == 1) & (test['preds'] < 0.1) & (test['servicer_message_time'] < TRUE_OVER_TIME) & (test['user_message_time'] < TRUE_OVER_TIME), 'preds'] = 0.9
    test.loc[(test['label'] == 0) & (test['preds'] > 0.9) & (test['servicer_message_time'] > FALSE_OVER_TIME) & (test['user_message_time'] > FALSE_OVER_TIME), 'preds'] = 0.1
    compute_metrics(test['label'], test['preds'], thresh=score_thresh, show=True)

    # print('\n------- 模型bridge效果 -------')
    # bridge = split_bridge_dataset(test, rt=0.2)
    # compute_metrics(bridge['label'], bridge['preds'], thresh=score_thresh, show=True)
