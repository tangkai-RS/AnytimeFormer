from model.layers import *
from model.utils import masked_mae_cal, total_variation_loss, masked_mape_cal


class AnytimeFormer(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.args = args
        self.learn_emb = args.learn_emb        
        self.with_X_aux = args.with_X_aux
        self.with_atten_mask = args.with_atten_mask
        self.n_group_inner_layers = args.n_group_inner_layers
        
        # loss settings
        self.with_rec_loss = args.with_rec_loss

        # Transformer Blocks
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    args.d_time,
                    args.d_feature,
                    args.d_model,
                    args.d_inner,
                    args.n_head,
                    args.d_k,
                    args.d_v,
                    args.dropout,
                    args.dropout,
                    args.diagonal_attention_mask
                )
                for _ in range(args.n_groups)
            ]
        )

        self.embedding_X = nn.Linear(args.d_feature, args.d_model, bias=False) # Optical embedding
        self.embedding_X_aux = EmbeddingSar(args.d_feature_aux, args.d_model) # SAR embedding
        self.position_enc = PositionalEncoding(args.d_model) # Time embedding
        self.reduce_dim = nn.Linear(args.d_model, args.d_feature)

        
        if self.args.with_X_aux:
            # Low-rank Fusion module
            self.tensorfusion = LowRankTensorFusion(args.d_model, args.rank)
            # Time-align attention module
            self.timealign_attention = MultiHeadAttention(
                args.n_head, args.d_model, args.d_k, args.d_v, args.dropout
            )
            self.multimodal_encoder = \
                MultiModalEncoderLayer(
                    args.d_time,
                    args.d_feature,
                    args.d_model,
                    args.d_inner,
                    args.n_head,
                    args.d_k,
                    args.d_v,
                    args.dropout,
                    args.dropout,
                    args.diagonal_attention_mask            
            )       
             
        # Time-aware Decoder
        self.decoder = DecoderLayer( 
            args.d_model, args.n_head,  args.d_k, args.d_v
        )
        
        if self.learn_emb:
            self.periodic = nn.Linear(1, args.d_model - 1)
            self.linear = nn.Linear(1, 1)
        
    def learn_time_embedding(self, dates):
        dates = dates.unsqueeze(-1)
        out2 = torch.sin(self.periodic(dates / 365. - 0.5))
        out1 = self.linear(dates / 365. - 0.5)
        return torch.cat([out1, out2], -1)    
    
    def _standscale(self, x):
        return (x - self.args.X_mean) / self.args.X_std
    
    def _reverse_standscale(self, x):
        return x * self.args.X_std + self.args.X_mean
    
    def anytime_mode(self, inputs, rec_data, valid_mask, stage):
        # For anytime prediction, only the train dates are used for rec and impute loss calculations
        # Other dates are regularized with Total Variation (TV) loss only    
        total_varitation_loss = torch.tensor(0.)    
        doy_start = torch.min(inputs["date_output"])
        # Anytime mode but not stage "anytime" -> only TV loss & gather train dates
        if ("anytime" in self.args.mode) and (stage != "anytime"):
            # cal TV loss
            if self.args.with_tv:
                total_varitation_loss = total_variation_loss(rec_data) 
            date_input_idx = (inputs["date_input"] - doy_start).long()       
            rec_data = rec_data.gather(
                1, date_input_idx.unsqueeze(-1).expand(-1, -1, rec_data.size(-1))
            )
            imputed_data = (1 - valid_mask) * rec_data + inputs["X"] * valid_mask
        # Stage "anytime" with daily outputs
        # len(self.args.anytime_ouput) == 0 represent daily outputs
        elif stage == "anytime" and len(self.args.anytime_ouput) == 0:
            date_input_idx = (inputs["date_input"] - doy_start).long()       
            rec_data_input = rec_data.gather(
                1, date_input_idx.unsqueeze(-1).expand(-1, -1, rec_data.size(-1))
            )
            imputed_data_input = (1 - valid_mask) * rec_data_input + inputs["X"] * valid_mask
            imputed_data = rec_data.clone()
            imputed_data.scatter_(
                1, date_input_idx.unsqueeze(-1).expand(-1, -1, rec_data.size(-1)),
                imputed_data_input
            )  
        # Specified output timestamps and interval
        elif stage == "anytime" and len(self.args.anytime_ouput) == 3:
            imputed_data = rec_data.clone()
        # Default fallback (if not any special anytime handling)
        else:
            # replace non-missing part with original data, missing (mask is 0)
            imputed_data = (1 - valid_mask) * rec_data + inputs["X"] * valid_mask
        return total_varitation_loss, imputed_data, rec_data
                    
    def impute(self, inputs, stage=None):
        # deal inputs
        date_input, date_output = inputs["date_input"], inputs["date_output"]
        X, masks = inputs["X"], inputs["attention_mask"]
        if self.with_atten_mask: 
            atten_mask = masks
        else:
            atten_mask = None
        # except original missing and artificial missing
        valid_mask = (1 - masks).unsqueeze(2) # missing is 0, valid is 1
        
        # embedding the X
        input_X = X
        input_X = self.embedding_X(input_X)
        # position (time, dates, DOY) embedding
        if self.learn_emb:
            position_X = self.learn_time_embedding(date_input)
            position_X_output = self.learn_time_embedding(date_output.long())
        else:
            position_X = self.position_enc(date_input.long())
            position_X_output = self.position_enc(date_output.long())
        enc_output = input_X + position_X

        # encoding the X
        for encoder_layer in self.encoder:
            for _ in range(self.n_group_inner_layers):
                enc_output, _ = encoder_layer(enc_output, atten_mask)
        
        # dynamic alignment of optical and SAR Data for Fusion
        if self.with_X_aux:
            X_aux = inputs["X_aux"]
            dates_aux = inputs["dates_aux"]
            input_X_aux, dates_aux = self.embedding_X_aux(X_aux, dates_aux)
            if self.learn_emb:
                position_X_aux = self.learn_time_embedding(dates_aux)
            else:
                position_X_aux = self.position_enc(dates_aux.long())
            # dates of X as Q, dates of SAR as K, X_aux_emb as V
            input_X_aux, _ = self.timealign_attention(position_X, position_X_aux, input_X_aux)
            # low-rank tensor fusion for X and X_aux
            input_X_aux = input_X_aux + position_X
            enc_output = self.tensorfusion(enc_output, input_X_aux)
            # further fusion multimodal by self-attention mechanism        
            enc_output, _ = self.multimodal_encoder(enc_output)       
                 
        # dates of output as Q, latent features as K, latent features as V 
        dec_output, _ = self.decoder(position_X_output, enc_output)
        # reduce the model's feature dim to original reflection bands
        rec_data = self.reduce_dim(dec_output)
        
        # anytime_mode 
        total_varitation_loss, imputed_data, rec_data = self.anytime_mode(
            inputs, rec_data, valid_mask, stage
        )
        return imputed_data, rec_data, total_varitation_loss
     
    def forward(self, inputs, stage=None):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, rec_data, total_varitation_loss = self.impute(inputs, stage)
        
        imputed_data = self._reverse_standscale(imputed_data)
        rec_data = self._reverse_standscale(rec_data)
        X = self._reverse_standscale(X)
        X_holdout = self._reverse_standscale(inputs["X_holdout"])
        
        if stage != "anytime": 
            reconstruction_MAE = masked_mae_cal(rec_data, X_holdout, masks)
            imputation_MAE = masked_mae_cal(
                rec_data, X_holdout, inputs["indicating_mask"]
            )
        else:
            reconstruction_MAE, imputation_MAE = torch.tensor(0.), torch.tensor(0.)
        
        total_loss = imputation_MAE + 0.01 * total_varitation_loss
        if self.with_rec_loss:
            total_loss = total_loss + 1. * reconstruction_MAE

        mape = torch.tensor(0.)
        if stage == "test" and (self.args.gap_mode == "continuous"):
            mape = masked_mape_cal(rec_data, X_holdout, masks)
        
        return {
            "imputed_data": imputed_data,
            "reconstructed_data": rec_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "total_varitation_loss": total_varitation_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
            "X_holdout": X_holdout,
            "total_loss": total_loss,
            "mape_loss": mape
        }