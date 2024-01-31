import os
import glob
import numpy as np
import logging
import itertools
import scipy
import math

ROOT_DIR = 'evaluation_experiment/mos_naturalness'
LABEL_DIR = os.path.join(ROOT_DIR, 'test_set')
RESULT_DIR = os.path.join(ROOT_DIR, 'result')

def _prep_label_list(label_path):
    '''
    正解ラベルの前処理
    '''

    with open(label_path, mode='r', encoding='utf-8') as f:
        lines = [l.rstrip() for l in f.readlines()]
        # ダミーの6文を抜く
        lines = lines[6:]
        
        text_name_order = list()
        emotion_order = list()
        condition_order = list()
        for line in lines:
            _, text_name, emotion, condition = line.split('\t')
            text_name_order.append(text_name)
            emotion_order.append(emotion)
            condition_order.append(condition)

        return text_name_order, emotion_order, condition_order

def main():
    condition_norm = {
        'Ground_Truth': 'Ground_Truth',
        'Embedding_Table': 'Embedding_Table',
        'xvector_Ground_Truth': 'xvector_Ground_Truth',
        'xvector_M-student_Converted': 'xvector_Student_Converted',
        'xvector_F-student_Converted': 'xvector_Student_Converted',
        'SSLmodel_Ground_Truth': 'SSLmodel_Ground_Truth',
        'SSLmodel_M-student_Converted': 'SSLmodel_Student_Converted',
        'SSLmodel_F-student_Converted': 'SSLmodel_Student_Converted'
    }
    emotion_list = ['Neutral', 'Happy', 'Sad']
    emotion2id = {'Neutral': 1, 'Happy': 2, 'Sad': 3}
    id2emotion = {v: k for k, v in emotion2id.items()}

    log_file = os.path.join(RESULT_DIR, 'result.log')
    logging.basicConfig(
        filename = log_file,
        filemode = 'w',
        encoding = 'utf-8',
        level = logging.INFO,
        format='%(message)s'
    )

    # 正解ラベルをまとめる
    label_path = os.path.join(LABEL_DIR, 'mos_naturalness.tsv')
    text_name_order, emotion_order, condition_order = _prep_label_list(label_path)

    mos_value_dict = dict()  # これに全員の回答を感情かつ条件ごとにまとめる
    for emotion in emotion_list:
        for condition in condition_norm.values():
            mos_value_dict[f'{emotion}_{condition}_value'] = list()  # 回答
            mos_value_dict[f'{emotion}_{condition}_count'] = 0       # カウント

    # logging.info('MOS-value per participant:')
    answer_files = sorted(glob.glob(os.path.join(RESULT_DIR, 'answer', '*.txt')))
    for answer_file in answer_files:
        answer_list = list()
        with open(answer_file, mode='r', encoding='utf-8') as f:
            lines = [s.rstrip() for s in f.readlines()]
        for line in lines:
            for char in line:
                answer_list.append(int(char))

        # 回答の入力ミスのチェック
        if len(condition_order) != len(answer_list):
            print(answer_list)
            raise ValueError(
                f'check {os.path.basename(answer_file)}, correct answer number is {len(condition_order)}, but number of this is {len(answer_list)}'
            )
        
        for emotion, condition_raw, answer in zip(emotion_order, condition_order, answer_list):
            mos_value_dict[f'{emotion}_{condition_norm[condition_raw]}_value'].append(answer)
            mos_value_dict[f'{emotion}_{condition_norm[condition_raw]}_count'] += 1

    # MOS値算出
    for emotion in emotion_list:
        logging.info(f'[Emotion] {emotion}')
        for condition in sorted(set(condition_norm.values())):
            value_list = mos_value_dict[f'{emotion}_{condition}_value']
            # 平均
            mean = np.mean(value_list)
            # 不偏分散
            var = np.var(value_list, ddof=1)
            length = len(value_list)
            # 自由度
            deg_of_freedom = length - 1
            bottom, up = scipy.stats.t.interval(0.95, deg_of_freedom, loc=mean, scale=math.sqrt(var/length))

            logging.info(f'{condition}\t{round(mean, 2)}\t(bottom: {round(mean-bottom, 2)}, up: {round(up-mean, 2)})')

    # 感情ごとに，条件間で対応のあるt検定
    logging.info('[Paired t-test] Significant difference:')
            
    for emotion in emotion_list:
        for condition_pair in itertools.combinations(sorted(set(condition_norm.values())), 2):
            c1, c2 =condition_pair
            statistic, pvalue = scipy.stats.ttest_rel(
                mos_value_dict[f'{emotion}_{c1}_value'],
                mos_value_dict[f'{emotion}_{c2}_value']
            )
            
            # 有意差あり
            if pvalue < 0.05:
                logging.info(f'{emotion}:  between \"{c1}\" and \"{c2}\" (p = {round(pvalue, 3)})')
    
    print(f'Result file: {log_file}')

if __name__ == '__main__':
    main()
