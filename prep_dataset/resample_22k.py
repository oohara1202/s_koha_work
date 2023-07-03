# 音声のサンプリングレートをSoXを用いて22050 [Hz]へ変換
# python prep_dataset/resample_22k.py /abelab/DB4/JSUT/jsut_ver1.1-p1
# python prep_dataset/resample_22k.py /abelab/DB4/JVS_Corpus/jvs_ver1
# python prep_dataset/resample_22k.py /abelab/DB4/STUDIES
# python prep_dataset/resample_22k.py /abelab/DB4/STUDIES-2

import os
import argparse
import shutil
import glob

def get_parser():
    parser = argparse.ArgumentParser(description='Resample to 22k.')
    parser.add_argument(
        'src_path',
        type=str,
        help='FULL Path of dataset (Source)',
    )
    return parser

def main():
    dst_dir = 'dataset'

    args = get_parser().parse_args()
    src_path = args.src_path
    assert os.path.exists(src_path)

    # 末尾に"_22k"を付ける
    if os.path.basename(src_path) == '':  # 入力の末尾が"/"の場合
        basename = os.path.basename(src_path[:-1])
    else:
        basename = os.path.basename(src_path)
    
    # 出力ディレクトリ
    dst_path = os.path.join(dst_dir, basename+'_22k')

    # 丸々コピー
    shutil.copytree(src_path, dst_path)
    print(f'Copied: {basename}')

    # wavファイルを収集
    wav_files = glob.glob(os.path.join(dst_path, '**/*.wav'), recursive=True)

    for wav_file in wav_files:
        wav_file = str(wav_file)
        # 変換元をリネーム
        wav_file_bak = wav_file.replace('.wav', '.wav.bak')
        os.rename(wav_file, wav_file_bak)
        # リサンプリング
        os.system(
            'sox {} -t wavpcm -b 16 {} gain -n -3 rate -h -I 22050 dither -s'.format(
                wav_file_bak, wav_file
            )
        )
        # 変換元を削除
        os.remove(wav_file_bak)

    print(f'Resampled: {basename}')

if __name__ == '__main__':
    main()
    