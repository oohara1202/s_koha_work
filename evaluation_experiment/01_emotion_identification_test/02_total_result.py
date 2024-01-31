import os
import glob
import numpy as np
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ROOT_DIR = 'evaluation_experiment/01_emotion_identification_test'
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
        
        emotion_order_subset = list()
        condition_order_subset = list()
        for line in lines:
            _, emotion, condition = line.split('\t')
            emotion_order_subset.append(emotion)
            condition_order_subset.append(condition)

        return emotion_order_subset, condition_order_subset

def main():
    condition_list = [
    'Embedding_table',
    'xvector_Teacher_ground_truth',
    'xvector_M-student_conversion',
    'xvector_F-student_conversion',
    'SSLmodel_Teacher_ground_truth',
    'SSLmodel_M-student_conversion',
    'SSLmodel_F-student_conversion'
    ]
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

    # 文ごとの正解ラベルをまとめる
    emotion_order = list()
    condition_order = list()
    label_list = sorted(glob.glob(os.path.join(LABEL_DIR, '*.tsv')))
    for label_path in label_list:
        emotion_order_subset, condition_order_subset = _prep_label_list(label_path)
        emotion_order.extend(emotion_order_subset)
        condition_order.extend(condition_order_subset)

    correct_with_condition = dict()  # これに全員の正解と回答をまとめあげる
    for condition in condition_list:
        correct_with_condition[f'{condition}_label'] = list()   # 条件ごとの正解
        correct_with_condition[f'{condition}_answer'] = list()  # 条件ごとの回答

    logging.info('Accuracy per participant:')
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
        
        for emotion, condition, answer in zip(emotion_order, condition_order, answer_list):
            correct_with_condition[f'{condition}_label'].append(emotion)
            correct_with_condition[f'{condition}_answer'].append(id2emotion[answer])

        # 参加者ごとにaccuracy算出
        accuracy = accuracy_score([emotion2id[emotion] for emotion in emotion_order], answer_list)
        accuracy_round = np.round(accuracy, 3)
        logging.info(f'{os.path.splitext(os.path.basename(answer_file))[0]}: {accuracy_round}')

    for condition in condition_list:
        logging.info(f'[Condition] {condition}')

        logging.info('Confusion matrix:')
        cm = confusion_matrix(
            y_true = correct_with_condition[f'{condition}_label'],
            y_pred = correct_with_condition[f'{condition}_answer'],
            labels = emotion_list
        )
        support_num = len(correct_with_condition[f'{condition}_label'])/3
        logging.info(np.round(cm/support_num, 3))
        logging.info('')
        
        logging.info('Classification report:')
        report = classification_report(
            y_true = correct_with_condition[f'{condition}_label'],
            y_pred = correct_with_condition[f'{condition}_answer'],
            labels = emotion_list,
            digits = 3
        )
        logging.info(report)

    print(f'Result file: {log_file}')

if __name__ == '__main__':
    main()
