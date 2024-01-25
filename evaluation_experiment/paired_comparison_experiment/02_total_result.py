import os
import glob
import numpy as np
import logging

ROOT_DIR = 'evaluation_experiment/paired_comparison_experiment'
LABEL_DIR = os.path.join(ROOT_DIR, 'test_set')
RESULT_DIR = os.path.join(ROOT_DIR, 'result')

def _prep_label_list(label_path):

    '''
    正解ラベルの前処理
    '''
    with open(label_path, mode='r', encoding='utf-8') as f:
        lines = [s.rstrip() for s in f.readlines()]
    # tsvファイルの処理
    pair_list = [line.split('\t')[1:] for line in lines]
    pair_list = pair_list[4:]  # ダミーを削除

    pair_count_dict = dict()  # 回答を格納するdict
    for pair in pair_list:
        # a>b というkeyで保存
        key = '{}>{}'.format(pair[0], pair[1])
        pair_count_dict[key] = 0

    return pair_list, pair_count_dict

def total_result(condition):
    '''
    回答の集計
    '''

    log_file = os.path.join(RESULT_DIR, f'result.log')
    logging.basicConfig(
        filename = log_file,
        filemode = 'w',
        encoding = 'utf-8',
        level = logging.INFO,
        format='%(message)s'
    )

    label_path = os.path.join(LABEL_DIR, f'{condition}.tsv')

    # ペアの順番リストと回答をカウントするdict
    pair_list, pair_count_dict = _prep_label_list(label_path)

    answer_files = sorted(glob.glob(os.path.join(RESULT_DIR, condition, '*.txt')))
    for answer_file in answer_files:
        with open(answer_file, mode='r', encoding='utf-8') as f:
            answer_list = list(f.read().rstrip())
            answer_list = [int(answer) for answer in answer_list]
            assert len(pair_list) == len(answer_list)

            for answer, pair in zip(answer_list, pair_list):
                if answer == 1:  # 左のほうが強いと回答
                    pair_count_dict[f'{pair[0]}>{pair[1]}'] += 1

    logging.info(f'[Condition] {condition}')
    for k, v in sorted(pair_count_dict.items()):
        logging.info(f'{k}\t{v}')

    print(f'Result file of \"{condition}\": {log_file}')

def main():
    # 埋め込み条件一覧を取得（e.g., xvector）
    condition_list = os.listdir(os.path.join(ROOT_DIR, 'speech_set'))
    condition_list.remove('conversion')  # これは例外
    for condition in condition_list:
        total_result(condition)

if __name__ == '__main__':
    main()
