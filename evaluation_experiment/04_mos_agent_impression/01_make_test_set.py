import os
import glob
import numpy as np
import random
import soundfile as sf
import re

INPUT_DIR = 'evaluation_experiment/04_mos_agent_impression/speech_set'
OUTPUT_DIR = 'evaluation_experiment/04_mos_agent_impression/out'

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

def generate_tset(emotion):
    '''
    テスト音声セットの生成
    '''

    # ダミーデータ数
    dummy_num = 4

    speech_dir = os.path.join(INPUT_DIR, emotion)

    # 無音秒数 [sec]
    pause_sec = 0.6

    # sin波の秒数
    sin_sec_short = 0.6
    sin_sec_long = 1.6
    f0 = 880.0  # 音高（880=ラ高）
    A = 0.1     # 音量

    # 何文ごとにsin波形を挿入するか
    sin_freaquency = 5

    # 組み合わせ
    pairs = list()

    # 生徒発話
    student_files = sorted(glob.glob(os.path.join(speech_dir, '*Student*.wav')))
    for student_file in student_files:
        bname = os.path.splitext(os.path.basename(student_file))[0]
        replace_bname = re.sub('[FM]Student', 'Teacher', bname)
        teacher_files = sorted(glob.glob(os.path.join(speech_dir, f'{replace_bname}*.wav')))
        assert len(teacher_files) == 2  # 生徒発話に対して応答音声が2種類あるか
        for teacher_file in teacher_files:
            assert os.path.exists(teacher_file)
            pairs.append([student_file, teacher_file])

    # 前半後半でそれぞれシャッフル
    first_pairs = random.sample(pairs, len(pairs))  # 前半
    last_pairs  = random.sample(pairs, len(pairs))  # 後半
    dummy_pairs = random.sample(pairs, dummy_num)   # ダミーファイル

    # この順番で音声が流れる
    test_pairs = dummy_pairs + first_pairs + last_pairs

    # 挿入音声の生成
    _, fs = sf.read(test_pairs[0][0], dtype='float32')
    sin_short = make_sin(sin_sec_short, A, fs, f0)
    sin_long = make_sin(sin_sec_long, A, fs, f0)
    pause = make_pause(pause_sec, fs)

    # 音声バッファ
    wav_buffer = []
    # ラベルリスト作成
    label_list = list()

    # 始端音の挿入
    wav_buffer.extend(pause)
    wav_buffer.extend(sin_long)
    wav_buffer.extend(pause)
    for i in range(len(test_pairs)):
        for j in range(len(test_pairs[i])): # ペア
            # 音声読み込み
            x, _ = sf.read(test_pairs[i][j], dtype='float32')

            # 音声の挿入
            wav_buffer.extend(x)
            # pauseの挿入
            if j%2 == 0: # ペア間
                wav_buffer.extend(pause)
            else:        # ペア終わり
                wav_buffer.extend(pause)
        
        # ペア終わり
        if(i%sin_freaquency==4):  # 区切りがよし
            wav_buffer.extend(sin_long)
            wav_buffer.extend(pause)
        else:
            wav_buffer.extend(sin_short)
            wav_buffer.extend(pause)

        condition = os.path.splitext(os.path.basename(test_pairs[i][1]))[0].split('_', 1)[1]
        row = f'{i+1}\t{os.path.basename(test_pairs[i][0])}\t{os.path.basename(test_pairs[i][1])}\t{condition}\n'
        label_list.append(row)
    # 終端音の挿入
    wav_buffer.extend(sin_long)

    # 音声出力
    wav_buffer = np.asarray(wav_buffer, dtype='float32')
    save_path = os.path.join(OUTPUT_DIR, f'{emotion}.wav')
    sf.write(save_path, wav_buffer, fs)
    print(f'Saved test set: {save_path}')

    # ラベルリスト出力
    save_path = os.path.join(OUTPUT_DIR, f'{emotion}.tsv')
    with open(save_path, mode='w', encoding='utf-8') as f:
        f.writelines(label_list)
    print(f'Saved label: {save_path}')

def main():
    # 感情条件一覧を取得
    emotion_list = os.listdir(INPUT_DIR)
    for emotion in emotion_list:
        generate_tset(emotion)

if __name__ == '__main__':
    main()
