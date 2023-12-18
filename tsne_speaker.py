# vcによる話者性の変化の確認
import os
import glob
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.manifold import TSNE

from abelab_utils.extract_xvector import ExtractXvector

class PlotTSNE:
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=1234)

        self.root_dir = 'evaluation_data'
        self.data_dir = os.path.join(self.root_dir, 'analysis_vc')
        self.exp_dir = 'exp'
        self.speaker_list = ['Teacher', 'M-student', 'F-student', 'M-student_converted', 'F-student_converted']
        self.speaker_color = {
            'Teacher': 'g',
            'M-student': 'b',
            'F-student': 'r',
            'M-student_converted': 'c',
            'F-student_converted': 'm'
        }
        
        self.extract_xvector = ExtractXvector()

    def run(self):
        xvector_list = list()
        speaker_hue = list()

        for speaker in self.speaker_list:
            xvector_path = os.path.join(self.data_dir, f'{speaker}_xvector.pkl')

            # x-vectorが保存されているなら読み込む
            if os.path.exists(xvector_path):
                with open(xvector_path, 'rb') as f:
                    xvector_speaker = pickle.load(f)

            # 保存されていないなら抽出後に保存
            else:
                xvector_speaker = dict()
                wavfiles = glob.glob(os.path.join(self.data_dir, speaker, '*.wav'))
                for wavfile in tqdm(wavfiles):
                    filename = os.path.basename(wavfile)
                    # ファイル名をキーにx-vector格納
                    xvector_speaker[filename] = self.extract_xvector(wavfile)
                # 話者ごとのx-vectorを保存
                with open(xvector_path, 'wb') as f:
                    pickle.dump(xvector_speaker, f)

            print(f'{speaker}: {len(xvector_speaker)} utterances')

            xvector_list.extend(xvector_speaker.values())
            speaker_hue.extend([speaker]*len(xvector_speaker))

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
        # 検証用にpng
        plt.savefig(
            os.path.join(self.exp_dir, 'tsne_speaker.png'),
            bbox_inches='tight',
        )
        # 本番用にpdf，transparent=Trueは動いてなさそう
        plt.savefig(
            os.path.join(self.exp_dir, 'tsne_speaker.pdf'),
            bbox_inches='tight',
            transparent=True,
        )

def main():
    tsne = PlotTSNE()
    tsne.run()

if __name__ == '__main__':
    main()
