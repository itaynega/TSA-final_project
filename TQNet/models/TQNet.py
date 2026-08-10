import torch
import torch.nn as nn

from layers.DampedTrend import DampedTrendInstanceNorm

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        # Ablation switches. Upstream hard-coded these to True and expected you to
        # edit the file between variants; they are now flags. The defaults reproduce
        # the published model exactly.
        self.use_tq = bool(getattr(configs, 'use_tq', 1))
        self.channel_aggre = bool(getattr(configs, 'channel_aggre', 1))

        # Arm A (J-11, report/prereg-improvement.md sec 3 "Arm A"): damped-trend
        # instance normalisation, behind a flag that defaults to off. With the
        # flag off nothing below this line runs and the forward pass is the
        # published one, unchanged -- see layers/DampedTrend.py for the
        # mechanism and for the two implementation choices sec 3 leaves open.
        # phi is a config value; it is NOT selected here (that is J-12, on
        # validation MSE at H=96 only).
        self.use_damped_trend = bool(getattr(configs, 'use_damped_trend', 0))
        self.damped_phi = float(getattr(configs, 'damped_phi', 0.9))
        if self.use_damped_trend:
            self.damped_trend = DampedTrendInstanceNorm(
                self.seq_len, self.pred_len, self.damped_phi
            )

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forward(self, x, cycle_index):

        # damped-trend detrend (Arm A). Off by default; when off, `x` reaches
        # the instance norm below exactly as it always did.
        if self.use_damped_trend:
            x, trend_slope = self.damped_trend.detrend(x)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # b,s,c -> b,c,s
        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        input = self.input_proj(x_input+channel_information)

        hidden = self.model(input)

        output = self.output_proj(hidden+input).permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        # damped-trend re-application (Arm A), after the de-normalisation and
        # therefore in the original data scale, which is the scale the slope
        # was fitted in.
        if self.use_damped_trend:
            output = self.damped_trend.retrend(output, trend_slope)

        return output


