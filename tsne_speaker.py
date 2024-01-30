# x-vectorを用いた変換音声の話者性の可視化
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.manifold import TSNE

from abelab_utils.extract_xvector import ExtractXvector

class PlotTSNE_Speaker:
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=0)

        self.data_dir = 'evaluation_data/analysis_vc'
        self.exp_dir = 'exp/tsne'
        self.dump_dir = os.path.join(self.exp_dir, 'dump')
        os.makedirs(self.dump_dir, exist_ok=True)

        self.speaker_list = ['Teacher', 'M-student', 'F-student', 'M-student_converted', 'F-student_converted']
        self.speaker_color = {
            'Teacher': 'g',
            'M-student': 'b',
            'F-student': 'r',
            'M-student_converted': 'c',
            'F-student_converted': 'm'
        }

    def run(self):
        xvector_list = list()
        speaker_hue = list()

        for speaker in self.speaker_list:
            xvector_path = os.path.join(self.dump_dir, f'{speaker}_xvector.pkl')

            # x-vectorが保存されているなら読み込む
            if os.path.exists(xvector_path):
                with open(xvector_path, 'rb') as f:
                    xvector_per_spk = pickle.load(f)

            # 保存されていないなら抽出後に保存
            else:
                extract_xvector = ExtractXvector()

                xvector_per_spk = dict()
                wavfiles = glob.glob(os.path.join(self.data_dir, speaker, '*.wav'))
                for wavfile in tqdm(wavfiles):
                    filename = os.path.basename(wavfile)
                    # ファイル名をキーにx-vector格納
                    xvector_per_spk[filename] = extract_xvector(wavfile)
                # 話者ごとのx-vectorを保存
                with open(xvector_path, 'wb') as f:
                    pickle.dump(xvector_per_spk, f)

            print(f'{speaker}: {len(xvector_per_spk)} utterances')

            xvector_list.extend(xvector_per_spk.values())
            speaker_hue.extend([speaker]*len(xvector_per_spk))

        xvector_stack = np.stack(xvector_list)

        xvector_reduced = self.tsne.fit_transform(xvector_stack)

        plt.figure(figsize=(6, 6))

        sns.scatterplot(
            x=xvector_reduced[:, 0],
            y=xvector_reduced[:, 1],
            hue=speaker_hue,
            hue_order=self.speaker_list,  # 凡例の順番
            palette=self.speaker_color,
            linewidth=0,  # 枠線を消す
        )

        plt.legend(
            bbox_to_anchor=(1, 1),  # 凡例の位置
            fontsize='x-large',
            title='Speaker',
            title_fontsize='x-large',
            markerscale=2.0,
        )
        plt.tick_params(
            length=0,                            # 目盛の長さをゼロ
            labelbottom=False, labelleft=False,  # 目盛のラベルを削除
        )

        fname = 'tsne_speaker'
        # 検証用にpng
        plt.savefig(
            os.path.join(self.exp_dir, f'{fname}.png'),
            bbox_inches='tight',
        )
        # 本番用にpdf，transparent=Trueは動いてなさそう
        plt.savefig(
            os.path.join(self.exp_dir, f'{fname}.pdf'),
            bbox_inches='tight',
            transparent=True,
        )

def main():
    tsne_speaker = PlotTSNE_Speaker()
    tsne_speaker.run()

if __name__ == '__main__':
    main()
