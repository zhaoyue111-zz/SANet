python train_new.py --organized-npz \
    --dataset luna25_organized \
    --data-root /mnt/afs2/data/luna25_organized \
    --annotation /mnt/afs2/data/luna25_organized/annotation.csv \
    --train-list /mnt/afs2/data/luna25_organized/train.txt \
    --val-list /mnt/afs2/data/luna25_organized/val.txt \
    --out-dir /mnt/afs2/code/SANet/train_output \
    --ckpt /mnt/afs2/code/SANet/train_pretrained_hardsamples_rcnn40_PN11_v4/model/best_rcnn.ckpt