import os
import numpy as np
import matplotlib.pyplot as plt

xvector_values = {
    'gt_strong': 1.353,
    'text1_strong': 0.879,
    'text2_strong': 2.397,
    'gt_weak': -0.083,
    'text1_weak': -2.770,
    'text2_weak': -2.87,
    'text1_et': -0.816,
    'text2_et': 2.019 
}

ssl_model_values = {
    'gt_strong': -0.092,
    'text1_strong': -0.626,
    'text2_strong': 2.230,
    'gt_weak': -0.714,
    'text1_weak': -2.282,
    'text2_weak': -0.676,
    'text1_et': -0.786,
    'text2_et': 2.877
}

filename2hue = {
    'gt_strong': 'Natural with Strong Emotion',
    'gt_weak': 'Natural with Weak Emotion',
    'text1_et': 'Embedding Table',
    'text1_strong': 'Proposed with Strong Emotion',
    'text1_weak': 'Proposed with Weak Emotion',
    'text2_et': 'Embedding Table',
    'text2_strong': 'Proposed with Strong Emotion',
    'text2_weak': 'Proposed with Weak Emotion'
}

ROOT_DIR = 'evaluation_experiment/paired_comparison_experiment'
RESULT_DIR = os.path.join(ROOT_DIR, 'result')

def _plot(condition, value_dict):
    filename_list = list()
    value_list = list()
    for k, v in value_dict.items():
        filename_list.append(k)
        value_list.append(v)

    plt.figure(figsize=(10, 0.5))

    xmin, xmax= -3.5, 3.5 # 数直線の最小値，最大値

    # 横軸の描写
    plt.hlines(
        y = 0,
        xmin = xmin,
        xmax = xmax,
        color = 'k'
    )  

    line_width = 1.0  # 目盛り数値の刻み幅

    # 目盛り線大
    plt.vlines(
        x = np.arange(xmin, xmax+line_width, line_width),
        ymin = -0.1,
        ymax = 0.1,
        color = 'k'
    )

    # 目盛り線小
    plt.vlines(
        x = np.delete(np.arange(xmin, xmax+line_width, line_width/2), -1),
        ymin = -0.05,
        ymax = 0.05,
        color = 'k'
    )

    # 目盛り数値
    plt.xticks(
        np.arange(xmin, xmax+line_width, line_width),
        fontsize = 12
    )

    # 枠線を消す
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # ラベルを消す
    plt.tick_params(
        labelleft = False,
        left = False,
        bottom =False
    )

    label_collection = list()
    for filename, value in zip(filename_list, value_list):
        # スーパーウルトラハードコーディングでマーカを設定
        linewidths = None
        ec = None
        if '_et' in filename:
            c = 'green'
            marker = 'o'
            y = 0.0
        elif 'strong' in filename:
            y = 0.02
            marker = ','
            if 'gt' in filename:
                c = 'w'
                ec = 'red'
                linewidths = 2
            else:
                c = 'red'
                marker = ','
        elif 'weak' in filename:
            y = -0.02
            marker = 'D'
            if 'gt' in filename:
                c = 'w'
                ec = 'blue'
                linewidths = 2
            else:
                c = 'blue'
            
        # 凡例のウルトラ対処療法
        if filename2hue[filename] not in label_collection:
            label = filename2hue[filename]
            label_collection.append(label)
        else:
            label = None

        plt.scatter(
            x = value,
            y = y,
            s = 200,
            label = label,
            c = c,
            marker = marker,
            linewidths=linewidths,
            ec=ec,
            alpha = 0.8
        )

    plt.legend(
        bbox_to_anchor=(0.675, -0.75),  # 凡例の位置
        markerscale = 0.70
    )

    # 検証用にpng
    plt.savefig(
        os.path.join(RESULT_DIR, f'vscale_{condition}.png'),
        bbox_inches='tight',
    )
    # 本番用にpdf，transparent=Trueは動いてなさそう
    plt.savefig(
        os.path.join(RESULT_DIR, f'vscale_{condition}.pdf'),
        bbox_inches='tight',
        transparent=True,
    )

def main():
    for condition in ['xvector', 'ssl_model']:
        exec(f'_plot(condition, {condition}_values)')

if __name__ == '__main__':
    main()
