export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=MDK-MoE
pred_len=96
for covar_loss in 0.02 
do
    for level in  15 21
    do
        for learning_rate in 0.0001 0.005
        do
            for batch_size in  16 64 32 8 
            do
                for threshold_ratio in 0.1
                do
                    for seq_len in 96 720 336 192 
                    do
                        for d_model in 32 64 128 256 512 1024 2048
                        do
                            python -u run_MDK_MoE.py \
                                --is_training 1 \
                                --root_path ./dataset/ \
                                --data_path exchange_rate.csv \
                                --model_id Exchange_$seq_len'_'96 \
                                --model $model_name \
                                --data custom \
                                --features M \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --enc_in 8 \
                                --level $level \
                                --batch_size $batch_size \
                                --des 'Exp' \
                                --lradj 'cosine'\
                                --learning_rate $learning_rate \
                                --threshold_ratio $threshold_ratio \
                                --covar_loss $covar_loss \
                                --loss_moe 1 \
                                --moving_avg 25 \
                                --dropout 0.1 \
                                --d_model $d_model \
                                --pos True \
                                --itr 1 >logs/LongForecasting/9.30/exchange/$model_name'_'Exchange_$seq_len'_'96'_'$level'_'$learning_rate'_'$batch_size'_'$threshold_ratio'_'$covar_loss'_'$d_model.log
                                sleep 1        
                        done
                    done
                done
            done
        done
    done
done

