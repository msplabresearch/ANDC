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
        
#         return linear_hs_proj_a1



# import torch
# from torch import nn
# import torch.nn.functional as F
# from modules.transformer import TransformerEncoder



# class SSL_MODEL(nn.Module):
#     def __init__(self, model_args):
#         """
#         """
#         super(SSL_MODEL, self).__init__()
#         self.d_a, self.d_v = 130, 130
#         self.hidden = 256
        
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
#         combined_dim = 2*self.d_a
        
#         # 1D convolutional projection layers
#         # self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)
#         # self.gru = nn.GRU(self.d_v, self.d_v, 2, batch_first=True, dropout=self.attn_dropout)
        
#         # Self Attentions 
#         self.a_mem = self.transformer_arch(self_type='audio_self')
#         self.v_mem = self.transformer_arch(self_type='visual_self')
        
#         self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar = True)
#         self.trans_v_mem = self.transformer_arch(self_type='visual_self', scalar = True)
        
#         # Cross-modal 
#         self.trans_v_with_a = self.transformer_arch(self_type='visual/audio', pos_emb = True)
#         self.trans_a_with_v = self.transformer_arch(self_type='audio/visual', pos_emb = True)
       
#         # Projection layers
#         self.proj_aux1 = nn.Linear(self.d_a, 512)
#         self.proj_aux2 = nn.Linear(512, 256)
#         self.proj_aux3 = nn.Linear(256, self.d_v)

#         # self.out_layer_au_2 = nn.Linear(self.d_v, 5)
#         # self.out_layer_landst_2 = nn.Linear(self.d_v, 1)
#         # self.out_layer_landsb_2 = nn.Linear(self.d_v, 1)
#         # self.out_layer_energy_2 = nn.Linear(self.d_v, 1)

#         # self.out_layer_aux = nn.Linear(self.d_v, output_dim)

        
#         # Projection layers
#         self.proj1 = nn.Linear(self.d_a, 512)
                
#         # # Projection layers
#         # self.proj2 = nn.Linear(512, 256)
#         # self.out_layer_au = nn.Linear(256, 5)

#         # self.proj3t = nn.Linear(512, 256)
#         # self.out_layer_landst = nn.Linear(256, 1)

#         # self.proj3b = nn.Linear(512, 256)
#         # self.out_layer_landsb = nn.Linear(256, 1)

#         # self.proj4 = nn.Linear(512, 256)
#         # self.out_layer_energy = nn.Linear(256, 1)


#     def transformer_arch(self, self_type='audio/visual', scalar = False, pos_emb = False):
#         if self_type == 'visual/audio':
#             embed_dim, attn_dropout = self.d_a, 0
#         elif self_type == 'audio/visual':
#             embed_dim, attn_dropout = self.d_v, 0
#         elif self_type == 'audio_self':
#             embed_dim, attn_dropout = self.d_a, self.attn_dropout    
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

#         proj_x_a = x_aud.permute(2, 0, 1)

 
  
#         # Audio/Visual
#         h_av = self.trans_a_with_v(proj_x_a, proj_x_a, proj_x_a)
#         h_as = self.trans_a_mem(h_av)
#         representation_audio = h_as[-1]

    
#         # Concatenating audiovisual representations
#         av_h_rep = representation_audio

#         #Main network output
#         linear_hs_proj_a1 = F.dropout(F.relu(self.proj1(av_h_rep)), p=self.out_dropout, training=self.training)



#         # # A residual block
#         # last_hs_proj1 = self.proj2(linear_hs_proj_a1)
#         # # last_hs_proj += last_hs
#         # output_au_1 = self.out_layer_au(last_hs_proj1)
        

#         # last_hs_proj2t = self.proj3t(linear_hs_proj_a1)
#         # # last_hs_proj += last_hs
#         # output_lands_t_1 = self.out_layer_landst(last_hs_proj2t)

#         # last_hs_proj2b = self.proj3b(linear_hs_proj_a1)
#         # # last_hs_proj += last_hs
#         # output_lands_b_1 = self.out_layer_landsb(last_hs_proj2b)


#         # last_hs_proj3 = self.proj4(linear_hs_proj_a1)
#         # # last_hs_proj += last_hs
#         # output_energy_1 = self.out_layer_energy(last_hs_proj3)
        
        

#         #Auxiliary audio network
#         h_a1 = self.a_mem(proj_x_a)
#         h_a2 = self.a_mem(h_a1)
#         h_a3 = self.a_mem(h_a2)
#         h_rep_a_aux = h_a3[-1]   
            
            
#         #Audio auxiliary network output
#         linear_hs_proj_a = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_a_aux)), p=self.out_dropout, training=self.training))), p=self.out_dropout, training=self.training))
#         linear_hs_proj_a += h_rep_a_aux

#         # output_au_2 = self.out_layer_au_2(linear_hs_proj_a)
#         # output_lands_t_2 = self.out_layer_landst_2(linear_hs_proj_a)
#         # output_lands_b_2 = self.out_layer_landsb_2(linear_hs_proj_a)
#         # output_energy_2 = self.out_layer_energy_2(linear_hs_proj_a)
        
        
#         return linear_hs_proj_a1, linear_hs_proj_a





import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder



class SSL_MODEL(nn.Module):
    def __init__(self, model_args):
        """
        """
        super(SSL_MODEL, self).__init__()
        self.d_a, self.d_v = 130, 130
        self.hidden = 256
        
        # Model Hyperparameters
        self.num_heads = model_args.num_heads
        self.layers = model_args.layers
        self.attn_mask = model_args.attn_mask
        output_dim = model_args.output_dim
        self.a_dim = model_args.a_dim
        self.attn_dropout = model_args.attn_dropout
        self.relu_dropout = model_args.relu_dropout
        self.res_dropout = model_args.res_dropout
        self.out_dropout = model_args.out_dropout
        self.embed_dropout = model_args.embed_dropout
        self.d_a, self.d_v = 130, 130
        combined_dim = 2*self.d_a
        
        # 1D convolutional projection layers
        # self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.gru = nn.GRU(self.d_v, self.d_v, 2, batch_first=True, dropout=self.attn_dropout)
        
        # Self Attentions 
        self.a_mem_1 = self.transformer_arch(self_type='audio_self')
        self.a_mem_2 = self.transformer_arch(self_type='audio_self')
        self.a_mem_3 = self.transformer_arch(self_type='audio_self')
        # self.v_mem = self.transformer_arch(self_type='visual_self')
        
        self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar = True)
        # self.trans_v_mem = self.transformer_arch(self_type='visual_self', scalar = True)
        
        # Cross-modal 
        self.trans_v_with_a = self.transformer_arch(self_type='visual/audio', pos_emb = True)
        self.trans_a_with_v = self.transformer_arch(self_type='audio/visual', pos_emb = True)
       
        # Projection layers
        self.proj_aux1 = nn.Linear(self.d_a, 512)
        self.proj_aux2 = nn.Linear(512, 256)
        # self.proj_aux3_1 = nn.Linear(256, self.d_v)
        # self.proj_aux3_2 = nn.Linear(256, self.d_v)
        # self.proj_aux3_3 = nn.Linear(256, self.d_v)
        # self.proj_aux3_4 = nn.Linear(256, self.d_v)


        # self.out_layer_au_2 = nn.Linear(self.d_v, 5)
        # self.out_layer_landst_2 = nn.Linear(self.d_v, 1)
        # self.out_layer_landsb_2 = nn.Linear(self.d_v, 1)
        # self.out_layer_energy_2 = nn.Linear(self.d_v, 1)

        # self.out_layer_aux = nn.Linear(self.d_v, output_dim)

        
        # Projection layers
        self.proj1 = nn.Linear(self.d_a, 512)
                
        # Projection layers
        # self.proj2 = nn.Linear(512, 256)
        # self.out_layer_au = nn.Linear(256, 5)

        # self.proj3t = nn.Linear(512, 256)
        # self.out_layer_landst = nn.Linear(256, 1)

        # self.proj3b = nn.Linear(512, 256)
        # self.out_layer_landsb = nn.Linear(256, 1)

        # self.proj4 = nn.Linear(512, 256)
        # self.out_layer_energy = nn.Linear(256, 1)


    def transformer_arch(self, self_type='audio/visual', scalar = False, pos_emb = False):
        if self_type == 'visual/audio':
            embed_dim, attn_dropout = self.d_a, 0
        elif self_type == 'audio/visual':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio_self':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout    
        elif self_type == 'visual_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Not a valid network")
        
        return TransformerEncoder(embed_dim = embed_dim,
                                  num_heads = self.num_heads,
                                  layers = self.layers,
                                  attn_dropout = attn_dropout,
                                  relu_dropout = self.relu_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask,
                                  scalar = scalar,
                                  pos_emb = pos_emb)
    
    
    def forward(self, x_aud):
        """
        audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        
        
        x_aud = x_aud.transpose(1, 2)

        proj_x_a = x_aud.permute(2, 0, 1)

 
  
        # Audio/Visual
        h_av = self.trans_a_with_v(proj_x_a, proj_x_a, proj_x_a)
        h_as = self.trans_a_mem(h_av)
        representation_audio = h_as[-1]

    
        # # Concatenating audiovisual representations
        # av_h_rep = representation_audio

        #Main network output
        linear_hs_proj_a1 = F.dropout(F.relu(self.proj1(representation_audio)), p=self.out_dropout, training=self.training)



        # # A residual block
        # last_hs_proj1 = self.proj2(linear_hs_proj_a1)
        # # last_hs_proj += last_hs
        # output_au_1 = self.out_layer_au(last_hs_proj1)
        

        # last_hs_proj2t = self.proj3t(linear_hs_proj_a1)
        # # last_hs_proj += last_hs
        # output_lands_t_1 = self.out_layer_landst(last_hs_proj2t)

        # last_hs_proj2b = self.proj3b(linear_hs_proj_a1)
        # # last_hs_proj += last_hs
        # output_lands_b_1 = self.out_layer_landsb(last_hs_proj2b)


        # last_hs_proj3 = self.proj4(linear_hs_proj_a1)
        # # last_hs_proj += last_hs
        # output_energy_1 = self.out_layer_energy(last_hs_proj3)
        
        

        #Auxiliary audio network
        h_a1 = self.a_mem_1(proj_x_a)
        h_a2 = self.a_mem_2(h_a1)
        h_a3 = self.a_mem_3(h_a2)
        h_rep_a_aux = h_a3[-1]   
            
            
        #Audio auxiliary network output
        linear_hs_proj_a = F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_a_aux)), p=self.out_dropout, training=self.training))), p=self.out_dropout, training=self.training)


        # last_au_2 = self.proj_aux3_1(linear_hs_proj_a)
        # output_au_2 = self.out_layer_au_2(last_au_2)

        # last_t_2 = self.proj_aux3_2(linear_hs_proj_a)
        # output_lands_t_2 = self.out_layer_landst_2(last_t_2)

        # last_b_2 = self.proj_aux3_3(linear_hs_proj_a)
        # output_lands_b_2 = self.out_layer_landsb_2(last_b_2)

        # last_energy_2 = self.proj_aux3_4(linear_hs_proj_a)
        # output_energy_2 = self.out_layer_energy_2(last_energy_2)
        
        
        return linear_hs_proj_a1, linear_hs_proj_a