# 生徒の発話と講師の応答の感情が同じものを羅列
import os
import glob

STUDIES_DIR = 'dataset/STUDIES_22k'
OUT_DIR = 'evaluation_experiment/03_mos_naturalness/text_option'

TEST_SUBSET = ['LD01', 'LD02', 'LD03', 'SD01', 'SD06', 'SD07', 'SD12']

def main():
    spkEn2spkJp = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
    spkJp2spkEn = {v: k for k, v in spkEn2spkJp.items()}
    emoEn2emoJp = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}
    emoJp2emoEn = {v: k for k, v in emoEn2emoJp.items()}

    # 書き込む内容
    write_lines_dict = dict()
    # 感情ごとに保存
    for emotion in emoEn2emoJp.values():
        write_lines_dict[emotion] = list()

    for type_name in ['Long_dialogue', 'Short_dialogue']:  # 対話文
        type_dir = os.path.join(STUDIES_DIR, type_name)
        
        for dname in TEST_SUBSET:
           
            d = os.path.join(type_dir, dname)
            
            # 台本テキスト
            txt_files = sorted(glob.glob(os.path.join(d, '**/*.txt'), recursive=True))
            for txt_file in txt_files:
                filename = os.path.splitext(os.path.basename(txt_file))[0]

                with open(txt_file, mode='r', encoding='utf-8') as f:
                    lines = [s.rstrip() for s in f.readlines()]

                for i in range(0, len(lines)-1, 2):
                    # 生徒から発話からスタートしない場合，めんどいのでcontinue
                    if lines[0].split('|')[0] == '講師':
                        # i += 1
                        continue

                    # 話者，感情，テキスト
                    spkJp_utter, emoJp_utter, text_utter, _ = lines[i].split('|')  # 話者
                    
                    # 対話終わりならcontinue
                    if i == len(lines)-1:
                        continue

                    spkJp_respon, emoJp_respon, text_respon, _ = lines[i+1].split('|')  # 話者

                    if emoJp_utter == emoJp_respon:
                        t = txt_file.replace('_22k', '')
                        write_line = f'{t}\nTurn:{((i//2)+1):02}\n{spkJp_utter}|{text_utter}\n{spkJp_respon}|{text_respon}\n'
                        write_lines_dict[emoEn2emoJp[emoJp_utter]].append(write_line)

    for emotion in emoEn2emoJp.values():
        write_filepath = os.path.join(OUT_DIR, emotion+'.txt')
        with open(write_filepath, mode='w', encoding='utf-8') as f:
            f.writelines(write_lines_dict[emotion] )
        
if __name__ == '__main__':
    main()
