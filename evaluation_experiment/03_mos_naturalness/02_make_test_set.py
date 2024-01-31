import os
import glob
import numpy as np
import random
import soundfile as sf

INPUT_DIR = 'evaluation_experiment/03_mos_naturalness/speech_set'
OUTPUT_DIR = 'evaluation_experiment/03_mos_naturalness/out'

condition_dict = {
    'Ground_Truth': 'Ground_Truth',
    'embedding_table': 'Embedding_Table',
    'xvector_TeacherGT': 'xvector_Ground_Truth',
    'xvector_MConverted': 'xvector_M-student_Converted',
    'xvector_FConverted': 'xvector_F-student_Converted',
    'ssl_TeacherGT': 'SSLmodel_Ground_Truth',
    'ssl_MConverted': 'SSLmodel_M-student_Converted',
    'ssl_FConverted': 'SSLmodel_F-student_Converted'
}

def make_pause(sec, fs):
    '''
    sec秒の無音を生成する
    '''

    points = int(fs * sec)
    rng = np.random.default_rng()
    pause = rng.uniform(-1.0e-10, 1.0e-10, points)

    return pause

def make_sin(sec, a, fs, f0):
    '''
    sin波を生成する

    Parameters
    ----------
    sec: float
        秒数
    a: float
        振幅（=音量）
    fs: int
        サンプリング周波数 [Hz]
    f0: float
        音の高さ
    '''

    swav=[]
    for n in np.arange(fs * sec):
        #サイン波を生成
        s = a * np.sin(2.0 * np.pi * f0 * n / fs)
        swav.append(s)

    swav = np.asarray(swav, dtype='float32')

    return swav

def generate_tset():
    '''
    テスト音声セットの生成
    '''

    # ダミーデータ数
    dummy_num = 6

    # 無音秒数 [sec]
    pause_sec = 0.4

    # sin波の秒数
    sin_sec_short = 0.4
    sin_sec_long = 1.2
    f0 = 880.0  # 音高（880=ラ高）
    A = 0.1     # 音量

    # 何文ごとにsin波形を挿入するか
    sin_freaquency = 5

    # 構成wavファイル
    wav_files = sorted(glob.glob(os.path.join(INPUT_DIR, '**/*.wav'), recursive=True))

    # 前半後半でそれぞれシャッフル
    first_files = random.sample(wav_files, len(wav_files))  # 前半
    last_files  = random.sample(wav_files, len(wav_files))  # 後半
    dummy_files = random.sample(wav_files, dummy_num)       # ダミーファイル

    # この順番で音声が流れる
    test_files = np.hstack((dummy_files, first_files, last_files))

    # 挿入音声の生成
    _, fs = sf.read(wav_files[0], dtype='float32')
    sin_short = make_sin(sin_sec_short, A, fs, f0)
    sin_long = make_sin(sin_sec_long, A, fs, f0)
    pause = make_pause(pause_sec, fs)

    # 音声バッファ
    wav_buffer = []
    # ラベルリスト作成
    label_list = list()

    for i, test_file in enumerate(test_files):
        assert os.path.exists(test_file)
        # 音声読み込み
        x, _ = sf.read(test_file, dtype='float32')

        if(i%sin_freaquency==0):
            # 長いsin波の挿入
            wav_buffer.extend(pause)
            wav_buffer.extend(sin_long)
            wav_buffer.extend(pause)
        else:
            # 短いsin波の挿入
            wav_buffer.extend(pause)
            wav_buffer.extend(sin_short)
            wav_buffer.extend(pause)

        wav_buffer.extend(x)

        bname = os.path.splitext(os.path.basename(test_file))[0]
        textname, emotion, condition = bname.split('_', 2)

        row = f'{i+1}\t{textname}\t{emotion}\t{condition_dict[condition]}\n'
        label_list.append(row)
    # 終端音の挿入
    wav_buffer.extend(sin_long)

    # 音声出力
    wav_buffer = np.asarray(wav_buffer, dtype='float32')
    save_path = os.path.join(OUTPUT_DIR, 'mos_naturalness.wav')
    sf.write(save_path, wav_buffer, fs)
    print(f'Saved test set: {save_path}')

    # ラベルリスト出力
    save_path = os.path.join(OUTPUT_DIR, 'mos_naturalness.tsv')
    with open(save_path, mode='w', encoding='utf-8') as f:
        f.writelines(label_list)
    print(f'Saved label: {save_path}')

def main():
    generate_tset()

if __name__ == '__main__':
    main()
