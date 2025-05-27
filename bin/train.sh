deepspeed --master_port=29506 --include="localhost:0" supervised_finetuning.py \
	--dataset_config_name trajs.data \
	--output_dir ../output/llama2_mbpp_rpsvr_20241201_peft/ \
	--model_type llama \
	--model_name_or_path ../models/llama2-7b-hf/ \
	--tokenizer_name_or_path ../models/llama2-7b-hf/ \
	--cache_dir ../output/llama2_mbpp_rpsvr_20241201_peft/
