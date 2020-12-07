# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.enc_pos_concat1x1, args.enc_pos_concat1x1_mode, args.enc_pos_concat1x1_bias,
                                                args.pose_concat1x1_init_mode)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.dec_pos_concat1x1, args.dec_pos_concat1x1_mode, args.dec_pos_concat1x1_bias,
                                                args.dec_pos_transv1, args.pose_concat1x1_init_mode)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enc_pos_concat1x1=False,
                 enc_pos_concat1x1_mode=0, enc_pos_concat1x1_bias=False,
                 pose_concat1x1_init_mode='eye'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.enc_pos_concat1x1 = enc_pos_concat1x1
        self.enc_pos_concat1x1_mode = enc_pos_concat1x1_mode
        
        if enc_pos_concat1x1 == True and enc_pos_concat1x1_mode == 0:
            self.self_attn_pos_trans = nn.Linear(512, 256, bias=enc_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans, mode=pose_concat1x1_init_mode)
        elif enc_pos_concat1x1 == True and enc_pos_concat1x1_mode == 1:
            self.self_attn_pos_trans_q = nn.Linear(512, 256, bias=enc_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans_q, mode=pose_concat1x1_init_mode)
            self.self_attn_pos_trans_k = nn.Linear(512, 256, bias=enc_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans_k, mode=pose_concat1x1_init_mode)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        if self.enc_pos_concat1x1 == True and self.enc_pos_concat1x1_mode == 0:
            cat_feat = torch.cat([src, pos], dim=2)
            # print(cat_feat.shape)
            q = k = self.self_attn_pos_trans(cat_feat)
        elif self.enc_pos_concat1x1 == True and self.enc_pos_concat1x1_mode == 1:
            cat_feat = torch.cat([src, pos], dim=2)
            q = self.self_attn_pos_trans_q(cat_feat)
            k = self.self_attn_pos_trans_k(cat_feat)
        else:
            q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, dec_pos_concat1x1=False,
                 dec_pos_concat1x1_mode=0, dec_pos_concat1x1_bias=False, dec_pos_transv1=False,
                 pose_concat1x1_init_mode='eye'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dec_pos_concat1x1 = dec_pos_concat1x1
        self.dec_pos_transv1 = dec_pos_transv1
        self.dec_pos_concat1x1_mode = dec_pos_concat1x1_mode
        # instead of directly add feature and positional embedding
        # we try to concat (256->512) + 1x1 (512->256) to fuse the feature and the positional embedding
        if dec_pos_concat1x1 == True and (dec_pos_concat1x1_mode == 0 or dec_pos_concat1x1_mode == 2):
            self.self_attn_pos_trans = nn.Linear(512, 256, bias=dec_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans, mode=pose_concat1x1_init_mode)
            self.cross_attn_pos_trans = nn.Linear(512, 256, bias=dec_pos_concat1x1_bias)
            _init_trans(self.cross_attn_pos_trans, mode=pose_concat1x1_init_mode)
        elif dec_pos_concat1x1 == True and dec_pos_concat1x1_mode == 1:
            self.self_attn_pos_trans_q = nn.Linear(512, 256, bias=dec_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans_q, mode=pose_concat1x1_init_mode)
            self.self_attn_pos_trans_k = nn.Linear(512, 256, bias=dec_pos_concat1x1_bias)
            _init_trans(self.self_attn_pos_trans_k, mode=pose_concat1x1_init_mode)
            self.cross_attn_pos_trans = nn.Linear(512, 256, bias=dec_pos_concat1x1_bias)
            _init_trans(self.cross_attn_pos_trans, mode=pose_concat1x1_init_mode)
        if dec_pos_concat1x1 == True and dec_pos_concat1x1_mode == 2:
            self.cross_attn_key_pos_trans = nn.Linear(512, 256, bias=False)
            _init_trans(self.cross_attn_key_pos_trans, mode=pose_concat1x1_init_mode)

        # if dec_pos_transv1:
        #     self.self_attn_pos_trans_post = nn.Linear(100, 100, bias=False)
        #     self.self_attn_pos_trans_post.weight.data.copy_(torch.eye(100))
        #     self.cross_attn_pos_trans_post = nn.Linear(100, 100, bias=False)
        #     self.cross_attn_pos_trans_post.weight.data.copy_(torch.eye(100))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.dec_pos_concat1x1 == True and (self.dec_pos_concat1x1_mode == 0 or self.dec_pos_concat1x1_mode == 2):
            cat_feat = torch.cat([tgt, query_pos], dim=2)
            # print(cat_feat.shape)
            q = k = self.self_attn_pos_trans(cat_feat)
        elif self.dec_pos_concat1x1 == True and self.dec_pos_concat1x1_mode == 1:
            cat_feat = torch.cat([tgt, query_pos], dim=2)
            q = self.self_attn_pos_trans_q(cat_feat)
            k = self.self_attn_pos_trans_k(cat_feat)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
        
        # if self.dec_pos_transv1:
        #     q = k = self.self_attn_pos_trans_post(q.T).T
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.dec_pos_concat1x1:
            query_fused = self.cross_attn_pos_trans(torch.cat([tgt, query_pos], dim=2))
        else:
            query_fused = self.with_pos_embed(tgt, query_pos)
        if self.dec_pos_concat1x1 == True and self.dec_pos_concat1x1_mode == 2:
            key_fused = self.cross_attn_key_pos_trans(torch.cat([memory, pos], dim=2))
        else:
            key_fused = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(query=query_fused,
                                   key=key_fused,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.dec_pos_concat1x1:
            q = k = self.self_attn_pos_trans(torch.cat([tgt2, query_pos], dim=2))
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.cross_attn_pos_trans(torch.cat([tgt2, query_pos], dim=2)) if self.dec_pos_concat1x1 else self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class ClsDecRegDecTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.enc_pos_concat1x1, args.enc_pos_concat1x1_mode, args.enc_pos_concat1x1_bias,
                                                args.pose_concat1x1_init_mode)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer_cls = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.dec_pos_concat1x1, args.dec_pos_concat1x1_mode, args.dec_pos_concat1x1_bias,
                                                args.dec_pos_transv1, args.pose_concat1x1_init_mode)
        self.decoder_cls = TransformerDecoder(decoder_layer_cls, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        decoder_layer_reg = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.dec_pos_concat1x1, args.dec_pos_concat1x1_mode, args.dec_pos_concat1x1_bias,
                                                args.dec_pos_transv1, args.pose_concat1x1_init_mode)
        self.decoder_reg = TransformerDecoder(decoder_layer_reg, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embeds, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed_cls = query_embeds[0].unsqueeze(1).repeat(1, bs, 1)
        query_embed_reg = query_embeds[1].unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt_cls = torch.zeros_like(query_embed_cls)
        tgt_reg = torch.zeros_like(query_embed_reg)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs_cls = self.decoder_cls(tgt_cls, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed_cls).transpose(1, 2)
        hs_reg = self.decoder_reg(tgt_reg, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed_reg).transpose(1, 2)
        hss = [hs_cls, hs_reg]
        return hss, memory.permute(1, 2, 0).view(bs, c, h, w)

        
class DisentangledV1Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.enc_pos_concat1x1, args.enc_pos_concat1x1_mode, args.enc_pos_concat1x1_bias,
                                                args.pose_concat1x1_init_mode)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_norm_cls = nn.LayerNorm(d_model)
        decoder_norm_reg = nn.LayerNorm(d_model // 4)

        decoder_layer_cls = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args.dec_pos_concat1x1, args.dec_pos_concat1x1_mode, args.dec_pos_concat1x1_bias,
                                                args.dec_pos_transv1, args.pose_concat1x1_init_mode)
        self.decoder_cls = TransformerDecoder(decoder_layer_cls, num_decoder_layers, decoder_norm_cls,
                                          return_intermediate=return_intermediate_dec)
        decoder_layers = []
        decoders = []
        for i in range(4):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model // 4, nhead // 4, dim_feedforward // 4,
                    dropout, activation, normalize_before, args.dec_pos_concat1x1,
                    args.dec_pos_concat1x1_mode, args.dec_pos_concat1x1_bias,
                    args.dec_pos_transv1, args.pose_concat1x1_init_mode)
            )
            decoders.append(
                TransformerDecoder(
                    decoder_layers[i], num_decoder_layers, decoder_norm_reg,
                    return_intermediate=return_intermediate_dec)
                    )
        self.decoders_reg = nn.ModuleList(decoders)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embeds, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)


        query_embed_cls = query_embeds[0].unsqueeze(1).repeat(1, bs, 1)
        query_embed_reg = query_embeds[1]
        mask = mask.flatten(1)

        tgt_cls = torch.zeros_like(query_embed_cls)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs_cls = self.decoder_cls(tgt_cls, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed_cls).transpose(1, 2)
        hss = []
        hss.append(hs_cls)
        
        query_embed_reg_list = query_embed_reg.split(query_embed_reg.shape[0] // 4, dim=0)
        query_embed_reg_list = [embed.unsqueeze(1).repeat(1, bs, 1) for embed in query_embed_reg_list]

        tgt_reg = torch.zeros_like(query_embed_reg_list[0])

        for i in range(4):
            hss.append(
                self.decoders_reg[i](
                    tgt_reg, memory, memory_key_padding_mask=mask,
                    pos=pos_embed, query_pos=query_embed_reg_list[i]).transpose(1, 2)
                )
        return hss, memory.permute(1, 2, 0).view(bs, c, h, w)
        


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
    )

def build_clsdec_regdec_transformer(args):
    return ClsDecRegDecTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
    )


def build_disentangled_v1_transformer(args):
    return DisentangledV1Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _init_trans(x, mode='eye'):
    assert x.weight.shape == torch.Size([256, 512])
    if mode == 'eye':
        x.weight.data.copy_(torch.cat([torch.eye(256), torch.eye(256)], dim=1))
    elif mode == 'normal':
        return
    elif mode == 'orthogonal':
        raise NotImplementedError('Orthogonal initialization is not supported yet')