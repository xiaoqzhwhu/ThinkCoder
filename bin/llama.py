from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words:list, tokenizer):
        self.keywords = [torch.LongTensor(tokenizer.encode(w)[1:]).to(device) for w in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if len(input_ids[0]) >= len(k) and torch.equal(input_ids[0][-len(k):], k):
                return True
        return False

    
class LlamaInterface:
    def __init__(self, modelpath, peftpath=None, add_lora=False):

        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = {}
    

        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.modelpath = modelpath
        if add_lora and peftpath:
            self.model = PeftModel.from_pretrained(
                self.model,
                peftpath,
                torch_dtype=torch.float16,
            )

            
    def llama(self, prompt, temperature=1, max_tokens=1000, stop=None, do_sample=False):
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt").to(device)
        stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer)]) if stop else None
        generated_ids = self.model.generate(
            input_ids=encoded_prompt["input_ids"],
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            early_stopping=False,
            num_return_sequences=1,
            temperature=temperature,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ).to(device)
        decoded_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=('###' in prompt))

        if stop:
            for ending in stop:
                if decoded_output[0].endswith(ending):
                    decoded_output[0] = decoded_output[0][:-len(ending)]
                    break

        return decoded_output[0]


    def generate_responses_from_llama(self, prompts, temperature=0.3, max_tokens=1000, n=1, stop=None, start_prompts=None, do_sample=True):
        from tqdm import tqdm

        responses = []

        if start_prompts is None:
            start_prompts = ['' for _ in prompts]

        for i in range(n):
            for p_i, prompt in tqdm(enumerate(prompts)):
                if isinstance(prompt, list):
                    prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False).strip()
                    start_prompt = start_prompts[p_i]
                    prompt += ' ' + start_prompt
                decoded_output = self.llama(prompt, temperature, max_tokens, do_sample=do_sample, stop=stop)
                if '[/INST]' in decoded_output:
                    decoded_output = decoded_output.split('[/INST]')[-1].strip()
                elif '<|eot_id|>' in decoded_output:
                    if len(decoded_output.split('<|eot_id|>')[-1].strip()) == 0:
                        decoded_output = decoded_output.split('<|eot_id|>')[-2].strip()
                    else:
                        decoded_output = decoded_output.split('<|eot_id|>')[-1].strip()
                else:
                    decoded_output = decoded_output[len(prompt):]
                responses.append(decoded_output)
        #print(responses)
        return responses
