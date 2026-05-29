python3 gator/scripts/relpose/finetune_gator.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/gator-small-classification-000/checkpoints/last.ckpt \
    --model-config.enc-emb-dim 384 \
    --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 \
    --model-config.dec-num-heads 6 \
    --exp-name gator-small-000

python3 gator/scripts/relpose/finetune_jigsaw.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/jigsaw-small-000/checkpoints/last.ckpt \
    --model-config.enc-emb-dim 384 \
    --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 \
    --model-config.dec-num-heads 6 \
    --exp-name jigsaw-small-000 \
    --opt-params.blr 3e-4

python3 gator/scripts/relpose/finetune_mae.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/mae-small-000/checkpoints/last.ckpt \
    --model-config.enc-emb-dim 384 \
    --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 \
    --model-config.dec-num-heads 6 \
    --exp-name mae-small-000

python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/croco-small-001/checkpoints/last.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')"  \
    --exp-name croco-small-000
