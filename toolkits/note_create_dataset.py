import time

MISSING_VALUE = -999
SEP_STR = '[SEP]'
LAST_CONTENT_NUM = 10


def chat_feature_pre_process(data, polling_time, writer=None):
    last_message_time = int(time.mktime(time.strptime(data['message_time'].iloc[-1], '%Y-%m-%d %H:%M:%S')))

    data = data[data['message_time'] <= polling_time]
    if len(data) == 0:
        return None

    dset = {
        'user': {'content': '', 'message_time': MISSING_VALUE, 'wait_time': MISSING_VALUE, 'message_cnt': 0,
                 'recovery_time': []},
        'servicer': {'content': '', 'message_time': MISSING_VALUE, 'wait_time': MISSING_VALUE, 'message_cnt': 0,
                     'recovery_time': []},
        'tol': {'content': [], 'time_line': [], 'send_role_list': []}
    }
    pre_send_role = MISSING_VALUE

    now = int(time.mktime(time.strptime(polling_time, '%Y-%m-%d %H:%M:%S')))
    intervene_time = int(time.mktime(time.strptime(data['intervene_time'].iloc[0], '%Y-%m-%d %H:%M:%S')))
    label = 1 if now >= last_message_time else 0
    serv_time = now - intervene_time

    # 每次轮询时的用户小二特征
    for i in range(data.shape[0]):
        message_time = int(time.mktime(time.strptime(data['message_time'].iloc[i], '%Y-%m-%d %H:%M:%S')))
        content = data['content'].iloc[i]
        send_role = int(data['send_role'].iloc[i])

        dset['tol']['content'].append(content)
        dset['tol']['time_line'].append(str(message_time - intervene_time))
        dset['tol']['send_role_list'].append(str(send_role))

        if send_role == 1 or send_role == 2:  # user
            dset['user']['content'] = content
            dset['user']['message_time'] = message_time - intervene_time
            dset['user']['wait_time'] = now - message_time
            dset['user']['message_cnt'] += 1
            dset['servicer']['wait_time'] = 0
            if pre_send_role == 3 and dset['servicer']['message_time'] != MISSING_VALUE:
                dset['user']['recovery_time'].append(
                    message_time - intervene_time - dset['servicer']['message_time'])
            pre_send_role = 1
        elif send_role == 3:  # servicer
            dset['servicer']['content'] = content
            dset['servicer']['message_time'] = message_time - intervene_time
            dset['servicer']['wait_time'] = now - message_time
            dset['servicer']['message_cnt'] += 1
            dset['user']['wait_time'] = 0
            if pre_send_role == 1 and dset['user']['message_time'] != MISSING_VALUE:
                dset['servicer']['recovery_time'].append(
                    message_time - intervene_time - dset['user']['message_time'])
            pre_send_role = 3
        else:
            pass

    dset['tol']['content'] = SEP_STR.join(dset['tol']['content'][-LAST_CONTENT_NUM:])
    dset['tol']['time_line'] = SEP_STR.join(dset['tol']['time_line'][-LAST_CONTENT_NUM:])
    dset['tol']['send_role_list'] = SEP_STR.join(dset['tol']['send_role_list'][-LAST_CONTENT_NUM:])

    chat_time_feats = {
        'user_message_time': dset['user']['message_time'],
        'user_wait_time': dset['user']['wait_time'],
        'user_message_cnt_real': dset['user']['message_cnt'],
        'user_recovery_time': np.mean(dset['user']['recovery_time']) if len(
            dset['user']['recovery_time']) != 0 else MISSING_VALUE,
        'servicer_message_time': dset['servicer']['message_time'],
        'servicer_wait_time': dset['servicer']['wait_time'],
        'servicer_message_cnt_real': dset['servicer']['message_cnt'],
        'servicer_recovery_time': np.mean(dset['servicer']['recovery_time']) if len(
            dset['servicer']['recovery_time']) != 0 else MISSING_VALUE,
        'send_role': pre_send_role,
        'polling': serv_time
    }
    chat_msg = {
        'last_user_content': dset['user']['content'].replace(',', '.').replace('\n', '.').replace('\r', '.'),
        'last_servicer_content': dset['servicer']['content'].replace(',', '.').replace('\n', '.').replace('\r', '.'),
        'content': dset['tol']['content'].replace(',', '.').replace('\n', '.').replace('\r', '.')
    }
    is_start_talking = pre_send_role != MISSING_VALUE
    res = [data['ds'].iloc[0], data['create_time'].iloc[0], data['intervene_time'].iloc[0], data['session_id'].iloc[0],
           chat_time_feats['polling'], polling_time,
           chat_msg['last_user_content'], chat_time_feats['user_message_time'], chat_time_feats['user_wait_time'],
           chat_time_feats['user_message_cnt_real'], chat_time_feats['user_recovery_time'],
           chat_msg['last_servicer_content'], chat_time_feats['servicer_message_time'],
           chat_time_feats['servicer_wait_time'], chat_time_feats['servicer_message_cnt_real'],
           chat_time_feats['servicer_recovery_time'],
           chat_msg['content'], chat_time_feats['time_line'], chat_time_feats['send_role_list'],
           chat_time_feats['send_role'], last_message_time - intervene_time, label]

    if writer is not None:
        writer.writerow(res)
    return res


import random


def make_ptimes(data, num=[4, 4, 2, 4], left=60, right=180):
    if len(data) == 0 or data['message_time'].iloc[0] == '' or data['message_time'].iloc[0] is None:
        return [], [], []
    intervene_time = int(time.mktime(time.strptime(data['intervene_time'].iloc[0], '%Y-%m-%d %H:%M:%S')))

    # 超时时间截断180s, 用户说话后60s无人说话的label=0样本上采样
    trunc = 180
    usr_upsampling_time = 60
    usr_upsampling = []

    before_time = intervene_time
    before_sr = -999
    lst_index = 0
    while lst_index < data.shape[0]:
        message_time = int(time.mktime(time.strptime(data['message_time'].iloc[lst_index], '%Y-%m-%d %H:%M:%S')))
        send_role = int(data['send_role'].iloc[lst_index])
        if message_time - before_time > trunc:
            break
        if before_sr == 1 and message_time - before_time >= usr_upsampling_time:
            usr_upsampling += [_ for _ in range(before_time + usr_upsampling_time, message_time - 1)]
        lst_index += 1
        before_time = message_time
        before_sr = send_role
    if lst_index == 0:
        return [], [], []
    data = data.drop(range(lst_index, len(data)))
    data.index = range(len(data))
    last_message_time = int(time.mktime(time.strptime(data['message_time'].iloc[-1], '%Y-%m-%d %H:%M:%S')))

    # 随机产生轮询时间
    # 4424 间隔3s
    #     if last_message_time - left - intervene_time > num[0] * 3:
    #         p0 = random.sample(range(intervene_time, last_message_time - left, 3), k=num[0])
    #         p1 = random.sample(range(last_message_time - left, last_message_time, 3), k=num[1])
    #     else:
    #         pt = random.sample(range(intervene_time, last_message_time, 3), k=min(num[0]+num[1], len(range(intervene_time, last_message_time, 3))))
    #         p0 = pt[:num[0]]
    #         p1 = pt[num[0]:] if len(pt) > num[0] else []
    #     p2 = random.sample(usr_upsampling, k=min(len(usr_upsampling), num[2]))
    #     p3 = random.sample(range(last_message_time, last_message_time+90, 3), k=num[3]-1)
    #     p3 += random.sample(range(last_message_time+90, last_message_time+right, 3), k=1)
    #     ptimes = p0 + p1 + p2 + p3

    # 间隔20s
    # if last_message_time + right - intervene_time > 20:
    #     ptimes = random.sample(range(intervene_time, last_message_time + right, 20),
    #                            k=min(len(range(intervene_time, last_message_time + right, 20)), sum(num)))
    #     p0, p1 = [], []
    #     p2 = [p for p in ptimes if p < last_message_time]
    #     p3 = [p for p in ptimes if p >= last_message_time]
    # else:
    #     return [], [], []

    # 间隔20s

    if random.random() > 0.5:
        ptimes = random.sample(range(intervene_time, last_message_time+right, 20),
                               k=min(len(range(intervene_time, last_message_time+right, 20)), sum(num)))
    else:
        ptimes = random.sample(range(intervene_time, last_message_time+90, 20),
                               k=min(len(range(intervene_time, last_message_time+90, 20)), sum(num)))
    p0 = [p for p in ptimes if p < last_message_time]
    p3 = [p for p in ptimes if p >= last_message_time]
    p1, p2 = [], []

    ptimes.sort()
    ptimes = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p)) for p in ptimes]
    return ptimes, data, (p0, p1, p2, p3)


import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import time
import csv

max_session_cnt = 400000
last_batch_data = []
ss_num = 0

columns = ['ds', 'session_id', 'user_id', 'user_wait_time', 'close_type_id', 'close_type_reason', 'create_time',
           'end_time', 'intervene_time', 'message_time',
           'content', 'send_role', 'question_id', 'answer_type', 'sop_id', 'sop_cate_id', 'sop_cate_name',
           'sop_cate_n_id', 'sop_cate_n_name', 'ord']

# 参考 pypai.io.TableReader 文档：https://aistudio.alipay.com/doc/odps_io.html
from pypai.io import TableReader

project_table_name = 'pai_temp_427063_12720853_1'
reader = TableReader.from_ODPS_type(o, project_table_name)
iterator = reader.to_iterator(batch_size=1000, columns=columns)  # columns不填默认读所有列

fp = open('dataset_sampling.csv', 'w', encoding='utf-8')
writer = csv.writer(fp)
writer.writerow(['ds', 'create_time', 'intervene_time', 'session_id', 'polling', 'polling_time',
                 'user_content', 'user_message_time', 'user_wait_time', 'user_message_cnt', 'user_recovery_time',
                 'servicer_content', 'servicer_message_time', 'servicer_wait_time', 'servicer_message_cnt',
                 'servicer_recovery_time',
                 'content', 'time_line', 'send_role_list', 'send_role', 'last_message_time', 'label'])
dist = [0] * 4
start_time = time.time()
while ss_num < max_session_cnt:
    try:
        batch_data = next(iterator)
        batch_data = last_batch_data + batch_data
        df = pd.DataFrame(batch_data, columns=columns)
        session_ids = np.unique(df['session_id'].values).tolist()
        last_sid = df['session_id'].iloc[-1]
        session_ids.remove(last_sid)
        last_batch_data = df[df['session_id'] == last_sid].values.tolist()
        for sid in session_ids:
            last_message_time = None
            data = df[df['session_id'] == sid]
            data.index = range(len(data))
            ptimes, data, _ = make_ptimes(data)
            if len(_) == 4:
                dist = [x + len(y) for x, y in zip(dist, _)]
            if len(data) == 0 or data['message_time'].iloc[0] == '' or data['message_time'].iloc[0] is None:
                continue
            for ptime in ptimes:
                feats = chat_feature_pre_process(data, ptime, writer=writer)

                if feats is None:
                    continue
                if last_message_time is None:
                    last_message_time = feats[-2]
                else:
                    if last_message_time != feats[-2]:
                        print('{} have many last_message_times {}/{}'.format(sid, last_message_time, feats[-2]))
            #                 if feats is not None:
            #                     features.append(feats)

            ss_num += 1
            if ss_num % 10000 == 0:
                print(
                    '{}|{} running... cost {:.2f}s. dist {}'.format(ss_num, max_session_cnt, (time.time() - start_time),
                                                                    np.array(dist) / ss_num))
                start_time = time.time()
    #             break
    #         break
    except:
        print('error:', traceback.print_exc())
        break
fp.close()