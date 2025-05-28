import torch
import torch.nn as nn
    
class fusion_model_c(torch.nn.Module):
    def __init__(self, image_feature_dim, mutimodal_feature_dim, projected_output_dim, nhead):
        super(fusion_model_c, self).__init__()
        
        if image_feature_dim % nhead != 0:
            raise ValueError(f"image_feature_dim ({image_feature_dim}) must be divisible by nhead ({nhead})")

        if mutimodal_feature_dim % nhead != 0:
            raise ValueError(f"mutimodal_feature_dim ({mutimodal_feature_dim}) must be divisible by nhead ({nhead})")
        
        image_encoder_layer = nn.TransformerEncoderLayer(
            d_model=image_feature_dim, 
            nhead=1, 
            dim_feedforward=image_feature_dim*4, 
            dropout=0.3, 
            batch_first=False
        )
        
        multimodal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=mutimodal_feature_dim, 
            nhead=2, 
            dim_feedforward=mutimodal_feature_dim*4, 
            dropout=0.3, 
            batch_first=False
        )
        
        # Transformer Encoder
        self.image_transformer_encoder = nn.TransformerEncoder(
            image_encoder_layer, 
            num_layers=1
        )
        
        self.multimodal_transformer_encoder = nn.TransformerEncoder(
            multimodal_encoder_layer, 
            num_layers=2
        )
        
        self.projection_fc = nn.Linear(mutimodal_feature_dim, projected_output_dim)


    def forward(self, text_features, image_features):
        text_tf = text_features.unsqueeze(0)
        image_tf = image_features.unsqueeze(0)        
        encoded_image_features = self.image_transformer_encoder(image_tf)
        
        # if text_features.shape[0] != encoded_image_features.shape[0] or \
        #    text_features.shape[1] != encoded_image_features.shape[1]:
        #     raise ValueError(
        #         f"Batch size and sequence length must match for text_features and encoded_image_features "
        #         f"to be concatenated along the feature dimension. "
        #         f"text_features shape: {text_features.shape}, "
        #         f"encoded_image_features shape: {encoded_image_features.shape}"
        #     )
            
        concatenated_features = torch.cat((text_tf, encoded_image_features), dim=2)
        fusion_features = self.multimodal_transformer_encoder(concatenated_features)
        fusion_features = self.projection_fc(fusion_features)
        return fusion_features.squeeze(0)
    
    
if __name__ == '__main__':
    text_features = torch.randn(26, 512)  # Example text features (batch_size, seq_length, feature_dim)
    image_features = torch.randn(26, 512)  # Example image features (batch_size, seq_length, feature_dim)
    model = fusion_model_c(image_feature_dim=512, mutimodal_feature_dim=1024, projected_output_dim=512, nhead=4)
    output = model(text_features, image_features)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_length, feature_dim * 2)
