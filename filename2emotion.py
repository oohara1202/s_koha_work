# ファイル名をキーに感情を返すdictを作る
# 対象はSTUDIESとCALLS（STUDIES-2）
import os
import glob
import pickle

def _get_filename2emotion_studies(studies_dir):
    enSpk2jpSpk = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
    enEmo2jpEmo = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}

    filename2emotion = dict()

    for type_name in ['ITA', 'Long_dialogue', 'Short_dialogue']:
        type_dir = os.path.join(studies_dir, type_name)
        
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]
        for dname in dir_list:
            d = os.path.join(type_dir, dname)

            for spk in ['Teacher', 'MStudent', 'FStudent']:
                txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
                wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*{spk}*.wav'), recursive=True))
                i = 0
                for txt_file in txt_files:
                    with open(txt_file, mode='r', encoding='utf-8') as f:
                        lines = f.readlines()
                    lines = [s for s in lines if s.split('|')[0]==enSpk2jpSpk[spk]]
                    for line in lines:
                        emotion = line.split('|')[1]  # 感情

                        filepath = wav_files[i]
                        filename = os.path.basename(filepath)

                        filename2emotion[filename] = enEmo2jpEmo[emotion]

                        i+=1

    return filename2emotion

def _get_filename2emotion_calls(calls_dir):
    enEmo2jpEmo = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}

    filename2emotion = dict()

    for type_name in ['P-dialogue', 'S-dialogue']:
        type_dir = os.path.join(calls_dir, type_name)
        
        # P01, S01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]
        for dname in dir_list:
            d = os.path.join(type_dir, dname)

            txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
            wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*.wav'), recursive=True))
            i = 0
            for txt_file in txt_files:
                with open(txt_file, mode='r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [s for s in lines if s.split('|')[0]=='オペレータ']
                for line in lines:
                    emotion = line.split('|')[1]  # 感情

                    filepath = wav_files[i]

                    filename2emotion[filepath] = enEmo2jpEmo[emotion]

                    i+=1

    return filename2emotion

def main():
    studies_dir = 'dataset/STUDIES_22k'
    calls_dir = 'dataset/STUDIES-2_22k'
    save_path = 'evaluation_data/filename2emotion.pkl'

    filename2emotion = _get_filename2emotion_studies(studies_dir)
    filename2emotion.update(_get_filename2emotion_calls(calls_dir))

    with open(save_path, 'wb') as f:
        pickle.dump(filename2emotion, f)

if __name__ == '__main__':
    main()
