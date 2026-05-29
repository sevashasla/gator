# ── Checkpoint directories ────────────────────────────────────────────────────
# CKPT_GATOR=/scratch/izar/skorokho/gator/relpose/gator-small-000-nf/checkpoints
# CKPT_CROCO_48HRS=/scratch/izar/bosi/gator/relpose/croco-small-48hrs-lr-1e-6-nf/checkpoints
# CKPT_CROCO_24HRS=/scratch/izar/bosi/gator/relpose/croco-small-24hrs-lr-1e-6-nf/checkpoints
# CKPT_JIGSAW=/scratch/izar/bosi/gator/relpose/jigsaw-small-24hrs-lr-1e-5-nf/checkpoints
# CKPT_MAE=/scratch/izar/bosi/gator/relpose/mae-small-unfrozen-blr3e5/checkpoints
# ─────────────────────────────────────────────────────────────────────────────

# ── Visualization commands
#
# 1. RECALL CURVE — all models on one plot
#
# python -m gator.scripts.relpose.recall_curve \
#   --ckpts \
#     gator:$CKPT_GATOR/last.ckpt \
#     croco48:$CKPT_CROCO_48HRS/last.ckpt \
#     croco24:$CKPT_CROCO_24HRS/last.ckpt \
#     mae:$CKPT_MAE/last.ckpt \
#     jigsaw:$CKPT_JIGSAW/last.ckpt \
#   --out visuals/relpose/recall_curve.png
#
# 2. FRUSTUM QUALITATIVE — image pairs + camera frustums
#    --sort_by: random (default, most neutral) | best (lowest max(rerr,terr)) | best_terr | best_rerr | terr | rerr
#    --scene:   chess | fire | heads | office | pumpkin | redkitchen | stairs
#
# python -m gator.scripts.relpose.visualize_relpose \
#   --ckpt $CKPT_GATOR/last.ckpt --model_name "Gator" \
#   --scene chess --n_samples 6 --sort_by random \
#   --out visuals/relpose/gator_chess_best.png
#
# python -m gator.scripts.relpose.visualize_relpose \
#   --ckpt $CKPT_CROCO_48HRS/last.ckpt --model_name "CroCo" \
#   --scene chess --n_samples 6 --sort_by random \
#   --out visuals/relpose/croco_chess_best.png
#
# python -m gator.scripts.relpose.visualize_relpose \
#   --ckpt $CKPT_MAE/last.ckpt --model_name "MAE" \
#   --scene chess --n_samples 6 --sort_by random \
#   --out visuals/relpose/mae_chess_best.png
#
# python -m gator.scripts.relpose.visualize_relpose \
#   --ckpt $CKPT_JIGSAW/last.ckpt --model_name "Jigsaw" \
#   --scene chess --n_samples 6 --sort_by random \
#   --out visuals/relpose/jigsaw_chess_best.png
# ─────────────────────────────────────────────────────────────────────────────

# MAE relpose - unfrozen, blr=3e-5
nohup python3 gator/scripts/relpose/finetune_mae.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/final-ckpts/mae-small-000-24hrs.ckpt \
    --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 --model-config.dec-num-heads 6 \
    --no-freeze-encdec --opt-params.blr 3e-5 --exp-name mae-small-unfrozen-blr3e5 \
    > nohup_mae_blr3e5.log 2>&1 &

# MAE relpose - unfrozen, blr=5e-5
nohup python3 gator/scripts/relpose/finetune_mae.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/final-ckpts/mae-small-000-24hrs.ckpt \
    --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 --model-config.dec-num-heads 6 \
    --no-freeze-encdec --opt-params.blr 5e-5 --exp-name mae-small-unfrozen-blr5e5 \
    > nohup_mae_blr5e5.log 2>&1 &

# CroCo relpose - unfrozen, blr=1e-6, 48hrs ckpt
nohup python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/croco-small-001-longer/checkpoints/last.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --exp-name croco-small-48hrs-lr-1e-6-nf \
    --opt-params.blr 1e-6 --no-freeze-encdec \
    --output-dir /scratch/izar/bosi/gator/relpose/ \
    > nohup_relpose_croco_48hrs.log 2>&1 &

# CroCo relpose - unfrozen, blr=1e-6, 24hrs ckpt
nohup python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/croco-small-001/checkpoints/last.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --exp-name croco-small-24hrs-lr-1e-6-nf \
    --opt-params.blr 1e-6 --no-freeze-encdec \
    --output-dir /scratch/izar/bosi/gator/relpose/ \
    > nohup_relpose_croco_24hrs.log 2>&1 &

# CroCo relpose - unfrozen, blr=1e-4, 24hrs ckpt
nohup python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/croco-small-001/checkpoints/last.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --exp-name croco-small-24hrs-lr-1e-4-nf \
    --opt-params.blr 1e-4 --no-freeze-encdec \
    --output-dir /scratch/izar/bosi/gator/relpose/ \
    > nohup_relpose_croco_24hrs.log 2>&1 &

# CroCo relpose - unfrozen, blr=1e-6, new ckpt lr=1e-4 (lowest val loss)
nohup python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/final-ckpts/croco-small-lr-1e-4-24hrs.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --exp-name croco-small-lr-1e-4-24hrs-nf \
    --opt-params.blr 1e-6 --no-freeze-encdec \
    --output-dir /scratch/izar/bosi/gator/relpose/ \
    > nohup_relpose_croco_lr1e4.log 2>&1 &

# Jigsaw relpose - unfrozen, blr=1e-5
nohup python3 gator/scripts/relpose/finetune_jigsaw.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/final-ckpts/jigsaw-small-000-24hrs.ckpt \
    --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 --model-config.dec-num-heads 6 \
    --no-freeze-encdec --opt-params.blr 1e-5 \
    --exp-name jigsaw-small-24hrs-lr-1e-5-nf \
    --output-dir /scratch/izar/bosi/gator/relpose/ \
    > nohup_relpose_jigsaw.log 2>&1 &
