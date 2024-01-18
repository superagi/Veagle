"""
Requires Transformer 4.35 and above, implementation may change according the Mistral implementation
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM

from veagle.common.registry import registry
from veagle.models.blip2 import Blip2Base, disabled_train

@registry.register_model("pretrain_veagle_mistral")
class PretrainVeagleMistral(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "mistral7b": "configs/models/veagle_mistral7b.yaml",
    }
     
    def __init__(
        self,
        llm_model,
        vision_model_path,
        freeze_vit=True,
        max_txt_len=128,
        max_output_txt_len=256,
        qformer_text_input=True,
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # Vision Encoder
        model = AutoModelForCausalLM.from_pretrained(vision_model_path)
        self.visual_encoder = model.get_model().vision_model
        logging.info("Vision Encoder Initialised")
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            logging.info("Freeze Vision Encoder")

        # LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, truncation_side="left", trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        logging.info(f"LLM {llm_model} Model Initialised")
        
        for name, param in self.llm_model.named_parameters():
            logging.info("LLM Model Freezed")
            param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        
        # Vision Projection Layer 
        self.vision_projection_layer_1 = nn.Linear(self.visual_encoder.config.hidden_size, 1408)
        self.vision_projection_layer_2 = nn.Linear(1408, self.llm_model.config.hidden_size)
        logging.info("Vision Projection Layer Initialised")
        
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        image = samples["image"]
        
        image_features= self.visual_encoder(image) # [batch_size, 257, 1024]
        image_features = image_features.last_hidden_state[:, 1:] 
        add_feature_llm = self.vision_projection_layer_2(self.vision_projection_layer_1(image_features))
        atts_add_feature_llm = torch.ones(add_feature_llm.size()[:-1], dtype=torch.long).to(image.device)
        
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        
        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        empty_add_targets = (
            torch.ones(atts_add_feature_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        
        targets = torch.cat((empty_add_targets, targets), dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        inputs_embeds = torch.cat((add_feature_llm, inputs_embeds), dim=1)
        attention_mask = torch.cat((atts_add_feature_llm, llm_tokens['attention_mask']), dim=1)
        
        with self.maybe_autocast():
            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        llm_model = cfg.get("llm_model")
        vision_model = cfg.get("vision_model")
        
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        qformer_text_input = cfg.get("qformer_text_input", True)

        return cls(
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            vision_model_path=vision_model,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            qformer_text_input=qformer_text_input,
        )