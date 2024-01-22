import os
import glob
import numpy as np
import random
import soundfile as sf

INPUT_DIR = 'evaluation_experiment/emotion_identification_test/speech_set'
OUTPUT_DIR = 'evaluation_experiment/emotion_identification_test/out'

emotion_list = ['Happy', 'Neutral', 'Sad']
condition_dict = {
    'et': 'Embedding_table',
    'xtg': 'xvector_Teacher_ground_truth',
    'xmc': 'xvector_M-student_conversion',
    'xfc': 'xvector_F-student_conversion',
    'stg': 'SSLmodel_Teacher_ground_truth',
    'smc': 'SSLmodel_M-student_conversion',
    'sfc': 'SSLmodel_F-student_conversion'
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

def generate_tset(sentence):
    '''
    テスト音声セットの生成
    '''

    speech_dir = os.path.join(INPUT_DIR, sentence)

    # 無音秒数 [sec]
    pause_sec = 0.6

    # sin波の秒数
    sin_sec = 1.0
    f0 = 880.0  # 音高（880=ラ高）
    A = 0.1     # 音量

    # 何文ごとにsin波形を挿入するか
    sin_freaquency = 5

    # 構成wavファイル
    wav_files = sorted(glob.glob(os.path.join(speech_dir, '*.wav')))

    # 前半後半でそれぞれシャッフル
    first_files = random.sample(wav_files, len(wav_files))  # 前半
    last_files  = random.sample(wav_files, len(wav_files))  # 後半

    # ダミーファイルの処理
    dummy_files = list()
    for condition in ['et', 'stg']:  # ダミーはEmbedding tableとSSL modelのTeacher ground truth
        for emotion in emotion_list:
            f = os.path.join(speech_dir, f'{sentence}_{emotion}_{condition}.wav')
            dummy_files.append(f)

    # この順番で音声が流れる
    test_files = np.hstack((dummy_files, first_files, last_files))

    # 挿入音声の生成
    _, fs = sf.read(wav_files[0], dtype='float32')
    sin = make_sin(sin_sec, A, fs, f0)
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
            # sin波の挿入
            wav_buffer.extend(pause[:len(pause)//2])  # 半分の無音挿入
            wav_buffer.extend(sin)
            wav_buffer.extend(pause[:len(pause)//2])  # 半分の無音挿入
        else:
            # 無音区間の挿入
            wav_buffer.extend(pause)

        wav_buffer.extend(x)

        bname = os.path.splitext(os.path.basename(test_file))[0]
        _, emotion, condition = bname.split('_')

        row = f'{i+1}\t{emotion}\t{condition_dict[condition]}\n'
        label_list.append(row)
    # 終端音の挿入
    wav_buffer.extend(sin)

    # 音声出力
    wav_buffer = np.asarray(wav_buffer, dtype='float32')
    save_path = os.path.join(OUTPUT_DIR, f'{sentence}.wav')
    sf.write(save_path, wav_buffer, fs)
    print(f'Saved test set: {save_path}')

    # ラベルリスト出力
    save_path = os.path.join(OUTPUT_DIR, f'{sentence}.tsv')
    with open(save_path, mode='w', encoding='utf-8') as f:
        f.writelines(label_list)
    print(f'Saved label: {save_path}')

def main():
    # 文章ID一覧を取得（e.g., j06）
    sentence_list = os.listdir(INPUT_DIR)
    for sentence in sentence_list:
        generate_tset(sentence)

if __name__ == '__main__':
    main()
