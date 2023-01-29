from transformers import AutoConfig
from modeling_bart import BartModel
import torch
import torch as th
import torch.nn as nn
from src.modeling.diffusion.nn import (
    SiLU,
    linear,
    timestep_embedding,
)
import math


class TransformerNetModel_encoder_decoder(nn.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        init_pretrained,
        freeze_embeddings,
        use_pretrained_embeddings,
        dropout=0,
        use_checkpoint=False,
        num_heads=1,
        config=None,
        config_name="bert-base-uncased",
        vocab_size=None,
        logits_mode=1,
        encoder_layers = 6,
        decoder_layers = 6,
        load_ckpt=None,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.dropout = dropout
            # config.hidden_size = 512

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained = init_pretrained
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.config_name = config_name
        self.load_ckpt = load_ckpt

        if not self.init_pretrained:
            self.config.encoder_layers = encoder_layers
            self.config.decoder_layers = decoder_layers
            self.config.vocab_size = vocab_size
            self.config.encoder_attention_heads = num_heads
            self.config.decoder_attention_heads = num_heads
            self.config.d_model = in_channels
            self.config.encoder_ffn_dim = model_channels
            self.config.decoder_ffn_dim = model_channels
            self.embedding_dim = 128 #self.config.d_model // 4
            self.embed_scale = math.sqrt(self.embedding_dim) if self.config.scale_embedding else 1.0

        time_embed_dim = in_channels
        self.time_embed = nn.Sequential(
            linear(in_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.d_model),
        )


        self.build_xstart_predictor()
        self.build_input_output_projections()
        self.build_embeddings()

        self.LayerNorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if self.load_ckpt is not None:
            self.load_weight(self.load_ckpt)

    def get_embeds(self, input_ids):
        return self.input_transformers.decoder.embed_tokens(input_ids) * self.embed_scale

    def load_weight(self, path):

        self.load_state_dict(torch.load(path))
        print(f'weigth initialize from {path}')

    def build_xstart_predictor(self):
        if self.init_pretrained:

            temp_bart = BartModel.from_pretrained(self.config_name, config=self.config)
            self.input_transformers = temp_bart
        else:
            self.input_transformers = BartModel(self.config, self.embedding_dim)

    def build_input_output_projections(self):
        if self.in_channels != self.embedding_dim:
             # need to adapt the model to the embedding size
            self.input_up_proj_dec = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )
                
            self.input_up_proj_enc = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.config.d_model),
                    nn.Tanh(),
                    nn.Linear(self.config.d_model, self.config.d_model),
                )

            self.output_down_proj = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.embedding_dim),
            )
        else:
            self.input_up_proj = nn.Identity()
            self.output_down_proj = nn.Identity()


    def build_embeddings(self):

        self.lm_head = nn.Linear(self.embedding_dim, self.input_transformers.shared.weight.shape[0])

        with th.no_grad():
            self.lm_head.weight = self.input_transformers.shared.weight

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward_encoder(self, 
                input_ids = None,
                timesteps = None,
                attention_mask = None,
                decoder_inputs_embeds = None,
                decoder_attention_mask = None,
                self_conditions = None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        # decoder_inputs_embeds = self.input_transformers.decoder.embed_tokens(decoder_input_ids) * self.embed_scale
        if self_conditions is not None:
            
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim = -1)

        decoder_inputs_embeds = (
            self.input_up_proj_dec(decoder_inputs_embeds)
            + emb
        )
        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))
        
        encoder_hidden_states = self.input_transformers(
            input_ids = None,
            attention_mask=attention_mask,
            inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale),
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs, 
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        ).encoder_last_hidden_state
        
        return encoder_hidden_states

    def forward(self, 
                input_ids = None,
                timesteps = None,
                attention_mask = None,
                decoder_inputs_embeds = None,
                decoder_attention_mask = None,
                self_conditions = None,
                encoder_outputs=None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert encoder_outputs is None or input_ids is None
        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        if self_conditions is not None:
            
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim = -1)

        decoder_inputs_embeds = (
            self.input_up_proj_dec(decoder_inputs_embeds)
            + emb
        )
        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))
        
        input_trans_hidden_states = self.input_transformers(
            input_ids = None,
            attention_mask=attention_mask,
            inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale) if input_ids is not None else None,
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs, 
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs
        ).last_hidden_state
        
        h = self.output_down_proj(input_trans_hidden_states)

        return h
