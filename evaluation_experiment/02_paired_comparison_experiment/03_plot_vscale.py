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

ROOT_DIR = 'evaluation_experiment/02_paired_comparison_experiment'
RESULT_DIR = os.path.join(ROOT_DIR, 'result')

def _plot(condition, value_dict):
    filename_list = list()
    value_list = list()
    for k, v in value_dict.items():
        filename_list.append(k)
        value_list.append(v)

    plt.figure(figsize=(10, 0.75))

    xmin, xmax= -3.5, 3.5 # 数直線の最小値，最大値

    # 横軸の描写
    plt.hlines(
        y = 0,
        xmin = xmin,
        xmax = xmax,
        color = 'k',
        zorder = 0
    )  

    line_width = 1.0  # 目盛り数値の刻み幅

    # 目盛り線大
    plt.vlines(
        x = np.arange(xmin, xmax+line_width, line_width),
        ymin = -0.1,
        ymax = 0.1,
        color = 'k',
        zorder = 0
    )

    # 目盛り線小
    plt.vlines(
        x = np.delete(np.arange(xmin, xmax+line_width, line_width/2), -1),
        ymin = -0.05,
        ymax = 0.05,
        color = 'k',
        zorder = 0
    )

    # 目盛り数値
    plt.xticks(
        np.arange(xmin, xmax+line_width, line_width),
        fontsize = 9
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
        ec = None
        linewidths = None
        s = 400
        if '_et' in filename:
            c = 'green'
            marker = 'o'
            y = 0.0
            s = s * 1.5
        elif 'strong' in filename:
            marker = ','
            y = 0.01
            s = s * 1.1
            if 'gt' in filename:
                c = 'w'
                ec = 'red'
                linewidths = 2
            else:
                c = 'red'
        elif 'weak' in filename:
            marker = 'D'
            y = -0.01
            if 'gt' in filename:
                c = 'w'
                ec = 'blue'
                linewidths = 2
            else:
                c = 'blue'
            
        # 凡例のウルトラ対処療法
        # 一度出てきたものを出さない
        if filename2hue[filename] not in label_collection:
            label = filename2hue[filename]
            label_collection.append(label)
        else:
            label = None

        plt.scatter(
            x = value,
            y = y,
            s = s,
            label = label,
            c = c,
            marker = marker,
            linewidths=linewidths,
            ec=ec,
            alpha = 0.8,
            zorder = 9
        )

        # 中に番号書きたいな
        if 'gt_' not in filename:
            if '1' in filename:
                num = '$\mathbf{1}$'
            else:
                num = '$\mathbf{2}$'
        else:
            continue

        plt.scatter(
            x = value-0.01,
            y = y,
            s = 200,
            label = None,
            c = 'w',
            marker = num,
            linewidths = 0.5,
            ec = 'k',
            zorder = 9
        )

    plt.legend(
        bbox_to_anchor=(0.675, -0.75),  # 凡例の位置
        markerscale = 0.425
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
    # パワポで追記するためのsvg
    plt.savefig(
        os.path.join(RESULT_DIR, f'vscale_{condition}.svg'),
        bbox_inches='tight',
        transparent=True,
    )

def main():
    for condition in ['xvector', 'ssl_model']:
        exec(f'_plot(condition, {condition}_values)')

if __name__ == '__main__':
    main()
