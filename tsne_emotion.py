# vc前後の感情表現の分析（x-vector or SSL-model feature）
import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.manifold import TSNE

# models.UtteranceLevelFeaturizerに基づく
class UtteranceLevelFeaturizer(nn.Module):
  def __init__(self, ssl_weights_path, layer_num=13):
    super().__init__()
    self.layer_num = layer_num
    self.weights = torch.load(ssl_weights_path)  # 
    print(f'Use weighted-sum weights:{self.weights}')

  def _weighted_sum(self, embeds_ssl):  # [batch(1), layer(13), frame(any), feature(768)]
    embeds_ssl = embeds_ssl.transpose(0, 1)  # [layer(13), batch(1), frame(any), feature(768)]
    _, *origin_shape = embeds_ssl.shape
    embeds_ssl = embeds_ssl.contiguous().view(self.layer_num, -1)  # [layer(13), any(any)]
    norm_weights = F.softmax(self.weights, dim=-1)
    weighted_embeds_ssl = (norm_weights.unsqueeze(-1) * embeds_ssl).sum(dim=0)  # [any(any)]
    weighted_embeds_ssl = weighted_embeds_ssl.view(*origin_shape)  # [batch(1), frame(any), feature(768)]
   
    return weighted_embeds_ssl

  def forward(self, embeds_ssl):
    embeds_ssl_BxTxH = self._weighted_sum(embeds_ssl)
    
    # averaged pooling
    embeds_ssl_TxH = embeds_ssl_BxTxH.squeeze()
    embeds_ssl = torch.mean(embeds_ssl_TxH, dim=0)

    return embeds_ssl

class PlotTSNE_Emotion:
    def __init__(self, feature_name):
        self.tsne = TSNE(n_components=2, random_state=0)

        self.feature_name = feature_name

        self.data_dir = 'evaluation_data/analysis_vc'
        self.exp_dir = 'exp/tsne'
        self.dump_dir = os.path.join(self.exp_dir, 'dump')
        os.makedirs(self.dump_dir, exist_ok=True)

        self.speaker_list = ['M-student', 'F-student']
        self.emotion_color = {
            'Neutral':'y',
            'Happy':'r',
            'Sad':'b'
        }
        
        with open('evaluation_data/filename2emotion.pkl', 'rb') as f:
            self.filename2emotion = pickle.load(f)

        if self.feature_name == 'xvector':
            from abelab_utils.extract_xvector import ExtractXvector
            self.extractor = ExtractXvector()
        elif self.feature_name == 'sslfeature':
            from abelab_utils.extract_sslfeature import ExtractSSLModelFeature
            self.extractor = ExtractSSLModelFeature()
            ssl_weights_path = '/work/abelab4/s_koha/vits/dump/averaged_vector_studies-teacher/weighted_sum_weights.pt'
            self.ulf = UtteranceLevelFeaturizer(ssl_weights_path)

    def _get_tsne_reduced(self, speaker):
        feature_list = list()
        emotion_hue = list()

        feature_path = os.path.join(self.dump_dir, f'{speaker}_{self.feature_name}.pkl')

        # 特徴量が保存されているなら読み込む
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                feature_per_spk = pickle.load(f)

        # 保存されていないなら抽出後に保存
        else:
            feature_per_spk = dict()
            wavfiles = glob.glob(os.path.join(self.data_dir, speaker, '*.wav'))
            for wavfile in tqdm(wavfiles):
                filename = os.path.basename(wavfile)
                # ファイル名をキーに特徴量格納
                # SSL
                if self.feature_name == 'sslfeature':
                    embeds_ssl = self.extractor(wavfile)
                    # [batch(1), layer(13), frame(any), feature(768)]
                    embeds_ssl = embeds_ssl.unsqueeze(0)
                    with torch.no_grad():
                        # Utterance-levelへ
                        feature = self.ulf(embeds_ssl)
                    feature_per_spk[filename] = feature
                # x-vector
                else:
                    feature_per_spk[filename] = self.extractor(wavfile)
            # 話者ごとの特徴量を保存
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_per_spk, f)

        print(f'{speaker}: {len(feature_per_spk)} utterances')

        for k, v in feature_per_spk.items():
            feature_list.append(v)
            emotion_hue.append(self.filename2emotion[k])

        feature_stack = np.stack(feature_list)

        feature_reduced = self.tsne.fit_transform(feature_stack)

        return feature_reduced, emotion_hue

    def plot_emotion(self, speaker):
        # 変換前と変換後のための2次元配列
        feature_reduced = [list() for i in range(2)]
        emotion_hue = [list() for i in range(2)]

        gt_conv = [speaker, f'{speaker}_converted']
        subplot_title = ['Natural', 'Conversion']

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, spk in enumerate(gt_conv):
            feature_reduced[i], emotion_hue[i] = self._get_tsne_reduced(spk)

            sns.scatterplot(
                x=feature_reduced[i][:, 0],
                y=feature_reduced[i][:, 1],
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

        fname = f'tsne_emotion_{self.feature_name}_{speaker}'
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
        # svg
        plt.savefig(
            os.path.join(self.exp_dir, f'{fname}.svg'),
            bbox_inches='tight',
            transparent=True,
        )

    def run(self):
        for speaker in self.speaker_list:
            self.plot_emotion(speaker)

def _get_parser():
    parser = argparse.ArgumentParser(description='Feature name.')
    parser.add_argument(
        'feature_name',
        type=str,
        choices=['xvector', 'sslfeature'],
        help='Feature name',
    )
    return parser

def main():
    args = _get_parser().parse_args()

    tsne_emotion = PlotTSNE_Emotion(args.feature_name)
    tsne_emotion.run()

if __name__ == '__main__':
    main()
