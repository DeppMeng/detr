NCCL_SOCKET_IFNAME=ib0

# MASTER_IP=${AZ_BATCH_MASTER_NODE%:*} && echo MASTER_IP: ${MASTER_IP}
# MASTER_PORT=${AZ_BATCH_MASTER_NODE##*:} && echo MASTER_PORT: ${MASTER_PORT}
MASTER_IP=${MASTER_IP}
MASTER_PORT=1234
NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}
PER_NODE_GPU=8 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=${OMPI_COMM_WORLD_SIZE} && echo NUM_NODE: ${NUM_NODE}

script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

pip install termcolor --user
pip list
MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --resume auto \
    --epochs 50 \
    --lr_drop 40 \
    --sine_query_embedv4 \
    --coco_path ../data/coco \
    --output_dir output/$script_name