import os
import sys
from . import chunk
from . import ser
from transformers import Wav2Vec2Model, WavLMModel, HubertModel, Data2VecAudioModel
import torch
from torch import nn
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.getcwd())
import sg_utils

class ModelWrapper():
    def __init__(self, args, **kwargs):
        self.args = args

        self.device = args.device
        self.model_type = args.model_type

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.output_num = args.output_num
        self.lab_type = args.label_type
        self.lbl_learning  = args.label_learning

        self.lr = args.lr

        self.model_path = args.model_path
        
        # self.use_chunk=args.use_chunk
        self.chunk_hidden_dim = args.chunk_hidden_dim
        self.chunk_window = args.chunk_window
        self.chunk_num = args.chunk_num

        return


    def init_model(self):
        """
        Define model and load pretrained weights
        """
        assert self.model_type in [
            "wav2vec2", "hubert", "wavlm", "data2vec",
            "wav2vec2-base", "wav2vec2-large", "wav2vec2-large-robust",
            "hubert-base", "hubert-large",
            "wavlm-base", "wavlm-base-plus", "wavlm-large",
            "data2vec-base", "data2vec-large"], \
            print("Wrong model type")
        
        default_models={
            "wav2vec2": "wav2vec2-large-robust",
            "hubert": "hubert-large",
            "wavlm": "wavlm-large",
            "data2vec": "data2vec-large",
        }
        real_model_name = default_models.get(self.model_type, self.model_type)
        assert real_model_name not in ["wav2vec2", "hubert", "wavlm", "data2vec"], \
            print("Model name is not properly converted.\n \
                Current model_name:", real_model_name
            )
        
        root_model_type = real_model_name.split("-")[0]
        assert root_model_type in ["wav2vec2", "hubert", "wavlm", "data2vec"], \
            print("Can't specify the root model type\n \
                Current root_model_type:", root_model_type
            )

        arch_type = real_model_name.split("-")[1]
        assert arch_type in ["base", "large"], \
            print("Can't specify the architecture type\n \
                architecture_type:", arch_type
            )

        # If base model, set is_large to False
        if arch_type == "large":
            is_large = True 
        elif arch_type == "base":
            is_large = False 
        else: 
            raise ValueError
        print("Loading", real_model_name)


        #### Wav2vec2
        if root_model_type == "wav2vec2":
            """
            Additional settings
            - Freeze feature encoder (for all wav2vec2 models)
            - Prune top 12 transformer layers (for wav2vec2-large-robust)
            """
            self.wav2vec_model= Wav2Vec2Model.from_pretrained("facebook/"+real_model_name)
            self.wav2vec_model.freeze_feature_encoder()
            if real_model_name == "wav2vec2-large-robust":
                del self.wav2vec_model.encoder.layers[12:]



        elif root_model_type == "data2vec":
            """
            Additional settings
            - Freeze feature encoder (for all data2vec models)
            """
            if real_model_name == "data2vec-large":
                self.wav2vec_model= Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")
            elif real_model_name == "data2vec-base":
                self.wav2vec_model= Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
            self.wav2vec_model.freeze_feature_encoder()
            
        elif root_model_type == "hubert":
            """
            Additional settings
            - Freeze feature encoder (for all hubert models)
            """
            if real_model_name == "hubert-large":
                self.wav2vec_model= HubertModel.from_pretrained("facebook/hubert-large-ll60k")
            elif real_model_name == "hubert-base":
                self.wav2vec_model= HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.wav2vec_model.feature_extractor._freeze_parameters()
            
        elif root_model_type == "wavlm":
            """
            Additional settings
            - Freeze feature encoder (for all wavlm models)
            """
            self.wav2vec_model= WavLMModel.from_pretrained("microsoft/"+real_model_name)
            self.wav2vec_model.freeze_feature_encoder()
            
        idim = 1024 if is_large else 768
        self.ser_model = ser.HLD(
            idim,
            self.hidden_dim, 
            self.num_layers, 
            self.output_num,
            self.lab_type,
            self.lbl_learning, 
            p=0.5)

        # self.wav2vec_model = nn.DataParallel(self.wav2vec_model)

        self.wav2vec_model.to(self.device)
        self.ser_model.to(self.device)
        # print(self.wav2vec_model)
        # print('model print')
        # print(self.ser_model)


        self.model_type_list = ["head", "wav2vec"]
        # if self.use_chunk:
        #     self.enable_chunk_model(idim)
        #     self.model_type_list.append("chunk")

    # def enable_chunk_model(self, *args, **kwargs):
    #     # assert self.use_chunk == True
    #     print("Apply chunk-based segmentation")
    #     chunk_input_dim = args[0]
    #     self.chunk_model = chunk.LSTM_AttenVec(
    #         chunk_input_dim, 
    #         self.chunk_hidden_dim,
    #         window_size = self.chunk_window,
    #         chunk_num = self.chunk_num
    #     )
    #     self.chunk_model.to(self.device)
        
    def init_optimizer(self):
        """
        Define optimizer for pre-trained model
        """
        assert self.wav2vec_model is not None and self.ser_model is not None, \
            print("Model is not initialized")
        
        self.wav2vec_opt = optim.Adam(self.wav2vec_model.parameters(), lr=self.lr)
        self.ser_opt = optim.Adam(self.ser_model.parameters(), lr=self.lr)
        # if self.use_chunk:
            # self.chunk_opt = optim.Adam(self.chunk_model.parameters(), lr=self.lr)
        # self.scaler = GradScaler()
    
    def feed_forward(self, x, eval=False, **kwargs):
        """
        Feed forward the model
        """
        def __inference__(self, x, **kwargs):
            mask = kwargs.get("attention_mask", None)
            if self.model_type == "wav2vec1":
                z = self.wav2vec_model.feature_extractor(x)
                w2v = self.wav2vec_model.feature_aggregator(z)
            else:
                # print(x, 'xxxxxxxxxxxxxxxxxxxxxxx')
                w2v = self.wav2vec_model(x, attention_mask=mask).last_hidden_state
                # print(w2v, 'W2V OUTPUTTTT')
            # if self.use_chunk:
            #     h = sg_utils.DynamicChunkForAll(w2v, self.chunk_window, self.chunk_num, 1)
            #     h = self.chunk_model(h)
            # else:
            h = sg_utils.AverageAll(w2v)
                # print(h, 'AVERAGING')
            pred = self.ser_model(h)
            return pred
        
        if eval:
            with torch.no_grad():
                return __inference__(self, x, **kwargs)
        else:
            return __inference__(self, x, **kwargs)
    
    def backprop(self, total_loss):
        """
        Update the model given loss
        """
        self.wav2vec_opt.zero_grad()
        self.ser_opt.zero_grad()
        # print(total_loss)
        total_loss.backward()
        self.wav2vec_opt.step()
        self.ser_opt.step()

    def save_model(self, epoch):
        """
        Save the model for each epoch
        """
  
        torch.save(self.wav2vec_model.state_dict(), \
            os.path.join(self.model_path, 'final_model.pt'))
        torch.save(self.ser_model.state_dict(), \
            os.path.join(self.model_path, "final_head.pt"))
        # if self.use_chunk:
        #     torch.save(self.chunk_model.state_dict(), \
        #         os.path.join(self.model_path, "final_chunk.pt"))
    
    def save_final_model(self, min_epoch, remove_param=False):
        """
        Copy the given epoch model to the final model
            if remove_param is True, remove the param folder
        """
        
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_head.pt") + \
        " "+os.path.join(self.model_path, "final_head.pt"))
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_model.pt") + \
            " "+os.path.join(self.model_path, "final_wav2vec.pt"))
        # if self.use_chunk:
        #     os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_chunk.pt") + \
        #     " "+os.path.join(self.model_path, "final_chunk.pt"))

        if remove_param:
            os.system("rm -rf "+os.path.join(self.model_path, "param"))

    def set_eval(self):
        """
        Set the model to eval mode
        """
        self.wav2vec_model.eval()
        self.ser_model.eval()
        # if self.use_chunk:
        #     self.chunk_model.eval()
    def set_train(self):
        """
        Set the model to train mode
        """
        self.wav2vec_model.train()
        self.ser_model.train()
        # if self.use_chunk:
        #     self.chunk_model.train()

    def load_model(self, model_path):
        
        self.wav2vec_model.load_state_dict(torch.load(model_path+"/final_model.pt", map_location=torch.device(self.device)))
        self.ser_model.load_state_dict(torch.load(model_path+"/final_head.pt", map_location=torch.device(self.device)))
        # if self.use_chunk:
        #     self.chunk_model.load_state_dict(torch.load(model_path+"/final_chunk.pt"))
