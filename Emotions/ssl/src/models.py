# import torch
# from torch import nn
# import torch.nn.functional as F
# from modules.transformer import TransformerEncoder



# class SSL_MODEL(nn.Module):
#     def __init__(self, model_args):
#         """
#         """
#         super(SSL_MODEL, self).__init__()
        
#         # Model Hyperparameters
#         self.num_heads = model_args.num_heads
#         self.layers = model_args.layers
#         self.attn_mask = model_args.attn_mask
#         output_dim = model_args.output_dim
#         self.a_dim, self.v_dim = model_args.a_dim, model_args.v_dim
#         self.attn_dropout = model_args.attn_dropout
#         self.relu_dropout = model_args.relu_dropout
#         self.res_dropout = model_args.res_dropout
#         self.out_dropout = model_args.out_dropout
#         self.embed_dropout = model_args.embed_dropout
#         self.d_a, self.d_v = 130, 130
        
#         # 1D convolutional projection layers
#         # self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)
#         # self.gru = nn.GRU(self.d_v, self.d_v, 2, batch_first=True, dropout=self.attn_dropout)

#         # Self Attentions 
#         self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar = True)
        
#         # Cross-modal 
#         self.trans_a = self.transformer_arch(self_type='audio/visual', pos_emb = True)

#         # Projection layers
#         self.proj1 = nn.Linear(self.d_a, self.d_a)
#         self.proj2 = nn.Linear(self.d_a, self.d_a)
#         self.out_layer = nn.Linear(self.d_a, output_dim)


#     def transformer_arch(self, self_type='audio/visual', scalar = False, pos_emb = False):
#         if self_type == 'visual/audio':
#             embed_dim, attn_dropout = self.d_v, 0
#         elif self_type == 'audio/visual':
#             embed_dim, attn_dropout = self.d_v, 0
#         elif self_type == 'audio_self':
#             embed_dim, attn_dropout = self.d_v, self.attn_dropout    
#         elif self_type == 'visual_self':
#             embed_dim, attn_dropout = self.d_v, self.attn_dropout
#         else:
#             raise ValueError("Not a valid network")
        
#         return TransformerEncoder(embed_dim = embed_dim,
#                                   num_heads = self.num_heads,
#                                   layers = self.layers,
#                                   attn_dropout = attn_dropout,
#                                   relu_dropout = self.relu_dropout,
#                                   res_dropout = self.res_dropout,
#                                   embed_dropout = self.embed_dropout,
#                                   attn_mask = self.attn_mask,
#                                   scalar = scalar,
#                                   pos_emb = pos_emb)
    
    
#     def forward(self, x_aud):
#         """
#         audio, and vision should have dimension [batch_size, seq_len, n_features]
#         """     
        
#         x_aud = x_aud.transpose(1, 2)
       
#         # 1-D Convolution visual/audio features
#         proj_x_a = x_aud.permute(2, 0, 1)
  
#         # Audio/Visual
#         h_av = self.trans_a(proj_x_a, proj_x_a, proj_x_a)
#         h_as = self.trans_a_mem(h_av)
#         representation_audio = h_as[-1]

#         #Main network output
#         linear_hs_proj_av = self.proj2(F.dropout(F.relu(self.proj1(representation_audio)), p=self.out_dropout, training=self.training))
#         linear_hs_proj_av += representation_audio
#         output = self.out_layer(linear_hs_proj_av)
        
        
#         return output


























import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder



# class SSL_MODEL(nn.Module):
#     def __init__(self, model_args):
#         """
#         """
#         super(SSL_MODEL, self).__init__()
        
#         # Model Hyperparameters
#         self.num_heads = model_args.num_heads
#         self.layers = model_args.layers
#         self.attn_mask = model_args.attn_mask
#         output_dim = model_args.output_dim
#         self.a_dim = model_args.a_dim
#         self.attn_dropout = model_args.attn_dropout
#         self.relu_dropout = model_args.relu_dropout
#         self.res_dropout = model_args.res_dropout
#         self.out_dropout = model_args.out_dropout
#         self.embed_dropout = model_args.embed_dropout
#         self.d_a, self.d_v = 130, 130
        
#         # 1D convolutional projection layers
#         # self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)
#         # self.gru = nn.GRU(self.d_v, self.d_v, 2, batch_first=True, dropout=self.attn_dropout)

#         # Self Attentions 
#         self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar = True)
        
#         # Cross-modal 
#         self.trans_a = self.transformer_arch(self_type='audio/visual', pos_emb = True)

#         # Projection layers
#         self.proj1 = nn.Linear(self.d_a, 512)
                
#         # Projection layers
#         self.projection = nn.Linear(512, 256)
#         self.output = nn.Linear(256, output_dim)



#     def transformer_arch(self, self_type='audio/visual', scalar = False, pos_emb = False):
#         if self_type == 'visual/audio':
#             embed_dim, attn_dropout = self.d_v, 0
#         elif self_type == 'audio/visual':
#             embed_dim, attn_dropout = self.d_v, 0
#         elif self_type == 'audio_self':
#             embed_dim, attn_dropout = self.d_v, self.attn_dropout    
#         elif self_type == 'visual_self':
#             embed_dim, attn_dropout = self.d_v, self.attn_dropout
#         else:
#             raise ValueError("Not a valid network")
        
#         return TransformerEncoder(embed_dim = embed_dim,
#                                   num_heads = self.num_heads,
#                                   layers = self.layers,
#                                   attn_dropout = attn_dropout,
#                                   relu_dropout = self.relu_dropout,
#                                   res_dropout = self.res_dropout,
#                                   embed_dropout = self.embed_dropout,
#                                   attn_mask = self.attn_mask,
#                                   scalar = scalar,
#                                   pos_emb = pos_emb)
    
    
#     def forward(self, x_aud):
#         """
#         audio, and vision should have dimension [batch_size, seq_len, n_features]
#         """     
        
#         x_aud = x_aud.transpose(1, 2)
       
#         # 1-D Convolution visual/audio features
#         proj_x_a = x_aud.permute(2, 0, 1)
  
#         # Audio/Visual
#         h_av = self.trans_a(proj_x_a, proj_x_a, proj_x_a)
#         h_as = self.trans_a_mem(h_av)
#         representation_audio = h_as[-1]

#         #Main network output
#         linear_hs_proj_a1 = F.dropout(F.relu(self.proj1(representation_audio)), p=self.out_dropout, training=self.training)



#         # A residual block
#         last_ha_proj = self.projection(linear_hs_proj_a1)
#         # last_hs_proj += last_hs
#         output = self.output(last_ha_proj)
        
        
#         return output




















class SSL_MODEL(nn.Module):
    def __init__(self, model_p, model_settings):
        super(SSL_MODEL, self).__init__()
        
        self.d_a, self.d_v = 130, 130
        self.output_dim = model_settings.output_dim
        self.pre_trained = model_p


        # Projection layers
        self.projection_1 = nn.Linear(512, 256)
        self.output_1 = nn.Linear(256, self.output_dim)

        self.projection_2 = nn.Linear(256, 128)
        self.output_2 = nn.Linear(128, self.output_dim)


    def forward(self, x_a):

        linear_hs_proj_a1, linear_hs_proj_a2 = self.pre_trained(x_a)


        # A residual block
        last_ha_proj = self.projection_1(linear_hs_proj_a1)
        # last_hs_proj += last_hs
        output_1 = self.output_1(last_ha_proj)


        # A residual block
        last_ha_proj = self.projection_2(linear_hs_proj_a2)
        # last_hs_proj += last_hs
        output_2 = self.output_2(last_ha_proj)
        
        
        return output_1, output_2
