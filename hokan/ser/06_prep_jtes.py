import os
import glob
from typing import Iterable, List, Optional, Union

def main():
    ############################################
    # ここを変える
    jtes_dirname = 'jtes_v1.1'  
    TEST_NUM = 5  # testのデータ数（testがvalidationも兼ねる）
    dst_dirname = 'filelists/jtes'   # 出力ディレクトリ
    dst_filename = 'jtes_audio_eid'  # 出力ファイル名（一部）
    emotions = ['neu', 'joy', 'sad', 'ang']
    emotion2id = {'neu':0, 'joy':1, 'sad':2, 'ang':3}  # 感情-->ID
    # emotions = ['neu', 'joy', 'sad']
    # emotion2id = {'neu':0, 'joy':1, 'sad':2}  # 感情-->ID
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    jtes_dir = os.path.join(basedir, jtes_dirname, 'wav')
    assert os.path.isdir(jtes_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # f01, m02などのディレクトリ名を取得
    dir_list = [f for f in os.listdir(jtes_dir) if os.path.isdir(os.path.join(jtes_dir, f))]
    dir_list.sort()

    # 出力するファイルリスト
    filelist = dict()
    filelist['test'] = list()
    filelist['train'] = list()

    for dname in dir_list:  # 話者ごとに回す
        for emotion in emotions:
            d = os.path.join(jtes_dir, dname, emotion)
            files = list()  # 感情ごとに一旦保存

            wav_files = sorted(glob.glob(os.path.join(d, '*.wav')))

            for wav_file in wav_files:
                newline = f'{wav_file}|{emotion2id[emotion]}\n'
                files.append(newline)
            
            filelist['test'].extend(files[:TEST_NUM])
            filelist['train'].extend(files[TEST_NUM:])

    for t in ['test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

if __name__ == '__main__':
    main()
