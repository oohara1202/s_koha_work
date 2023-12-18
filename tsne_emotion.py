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
        self.speaker_list = ['M-student', 'F-student']
        self.emotion_color = {
            'Neutral':'y',
            'Happy':'r',
            'Sad':'b'
        }
        
        with open(os.path.join(self.root_dir, 'filename2emotion.pkl'), 'rb') as f:
            self.filename2emotion = pickle.load(f)

        self.extract_xvector = ExtractXvector()

    def _get_tsne_reduced(self, speaker):
        xvector_list = list()
        emotion_hue = list()

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

        for k, v in xvector_speaker.items():
            xvector_list.append(v)
            emotion_hue.append(self.filename2emotion[k])

        xvector_stack = np.stack(xvector_list)

        xvector_reduced = self.tsne.fit_transform(xvector_stack)

        return xvector_reduced, emotion_hue

    def plot_emotion(self, speaker):
        xvector_reduced = [list() for i in range(2)]
        emotion_hue = [list() for i in range(2)]

        gt_conv = [speaker, f'{speaker}_converted']
        subplot_title = ['Ground Truth', 'Conversion']

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, spk in enumerate(gt_conv):
            xvector_reduced[i], emotion_hue[i] = self._get_tsne_reduced(spk)

            sns.scatterplot(
                x=xvector_reduced[i][:, 0],
                y=xvector_reduced[i][:, 1],
                hue=emotion_hue[i],
                hue_order=self.emotion_color.keys(),  # 凡例の順番
                palette=self.emotion_color,
                linewidth=0,  # 枠線を消す
                ax=axes[i],
            )

            axes[i].set_title(
                subplot_title[i],
                fontsize='x-large',
            )
            axes[i].tick_params(
                length=0,                            # 目盛の長さをゼロ
                labelbottom=False, labelleft=False,  # 目盛のラベルを削除
            )
            axes[i].legend().set_visible(False)  # 凡例消し

        plt.legend(
            bbox_to_anchor=(0.55, 0),  # 凡例の位置
            ncol=3,
            fontsize='x-large',
            title='Emotion',
            title_fontsize='x-large',
            markerscale=2.0,
        )

        # 検証用にpng
        plt.savefig(
            os.path.join(self.exp_dir, f'tsne_emotion_{speaker}.png'),
            bbox_inches='tight',
        )
        # 本番用にpdf，transparent=Trueは動いてなさそう
        plt.savefig(
            os.path.join(self.exp_dir, f'tsne_emotion_{speaker}.pdf'),
            bbox_inches='tight',
            transparent=True,
        )

    def run(self):
        for speaker in self.speaker_list:
            self.plot_emotion(speaker)

def main():
    tsne = PlotTSNE()
    tsne.run()

if __name__ == '__main__':
    main()
