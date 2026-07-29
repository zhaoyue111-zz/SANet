sensetimemed-4090 --workspace-name workspace -f pt -r n5lp.nn.a80.4 \
    --container-image registry.cn-sh-01.sensecore.cn/libangtong/aicl-ytt-0709:20250709-16h43m35s \
    bash -c 'export PATH=PATH:/XXX && source activate cloud-ai-lab && \
    cd /mnt/afs2/code/SANet/debug/ && \
    torchrun --standalone --nproc_per_node=1 sleep.py'