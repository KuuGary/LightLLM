from math import sqrt

import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from layers.Embed import PatchEmbedding

import transformers
from peft import PeftModel, PeftConfig, LoraConfig

from layers.StandardNorm import Normalize

from torch.nn.utils import weight_norm

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = 4096
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.llama_config = LlamaConfig.from_pretrained('/scratch/ey69/jh3193/llama-7b')
        # self.llama_config = LlamaConfig.from_pretrained('../../llama-7b')
        self.llama_config.num_hidden_layers = configs.llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.llama = LlamaModel.from_pretrained(
            "/scratch/ey69/jh3193/llama-7b",
            #  '../llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=self.llama_config,
            load_in_4bit=True
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            "/scratch/ey69/jh3193/llama-7b",
            #  '../llama-7b',
            trust_remote_code=True,
            local_files_only=True
        )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llama.parameters():
            param.requires_grad = False

        self.llama = self.add_lora(self.llama, configs)

        self.dropout = nn.Dropout(configs.dropout)

        # Task-specific encoder
        self.patch_embedding = TCNEmbedding(
    input_size=512, hidden_size=configs.d_model, kernel_size=3, dropout=configs.dropout)

        self.word_embeddings = self.llama.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer_Gary(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
    
    def add_lora(self, model, configs):
        peft_config = LoraConfig(
            task_type="SEQ2SEQ",
            r=64,
            lora_alpha=64,
            lora_dropout=0.1
        )
        return PeftModel(model, peft_config)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        mean_values = torch.mean(x_enc, dim=1)
        std_dev_values = torch.std(x_enc, dim=1)
        x_np = x_enc.cpu().numpy()
        skewness_values = skew(x_np, axis=1)
        kurtosis_values = kurtosis(x_np, axis=1)
        # print(x_enc.shape)
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            
            mean_values_str = str(mean_values[b].tolist()[0])
            std_dev_values_str = str(std_dev_values[b].tolist()[0])
            skewness_values_str = str(skewness_values[b].tolist()[0])
            kurtosis_values_str = str(kurtosis_values[b].tolist()[0])

            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: The Power Output Measurements dataset comprises data from a 30-kW rooftop photovoltaic (PV) array, logged at a 1-minute frequency. The dataset providing detailed insight into the power generation patterns of the PV system."
                f"Task description: forecast the next 16 steps given the previous 512 steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llama.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        print(f"n_vars: {n_vars}, enc_out shape: {enc_out.shape}")
        # print(f"n_vars: {n_vars}, dec_out shape: {dec_out.shape}")
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]


        # # Dynamic adjust shape
        # dec_out = torch.reshape(
        #     dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.size(-1) // (n_vars * dec_out.shape[-2]))
        # )
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        # print(x_enc.size(-1)) # check input size

        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer_Gary(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_Gary, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

        self.adapter = nn.Linear(d_model, d_model)
        self.norm_layer = nn.LayerNorm(d_model)  # Added LayerNorm

        # Learnable parameters to adjust the weight of data and word_embedding
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)
        out = self.out_projection(out)

        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # Adjust the weight of data and prompt embeddings
        weighted_target = self.alpha * target_embedding
        weighted_source = self.beta * source_embedding

        scores = torch.einsum("blhe,she->bhls", weighted_target, weighted_source)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class TCNEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, stride=8, padding=1, dropout=0.2, dilation=1):
        super(TCNEmbedding, self).__init__()
        self.causal_conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.causal_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, time_steps)
        x = self.causal_conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.causal_conv2(x)

        x = x.permute(0, 2, 1)

        n_vars = 1
        return x, n_vars

