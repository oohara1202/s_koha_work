SRC_DIR=./input_wav
DST_DIR=./output_wav

(cd ${SRC_DIR}; find * -type d) | xargs -I _DIR_ mkdir -p ${DST_DIR}/_DIR_

for f in $(cd ${SRC_DIR}; find * -name "*.wav" ); do
    f=$(dirname $f)/$(basename $f .wav)
    sox ${SRC_DIR}/$f.wav ${DST_DIR}/$f.wav norm
    echo ${SRC_DIR}/$f.wav ${DST_DIR}/$f.wav
done
