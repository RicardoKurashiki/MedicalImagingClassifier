import os
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class _Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class _Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, dropout=0.2):
        super(_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _Decoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def save_weights(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        torch.save(
            self.encoder.state_dict(), os.path.join(output_path, "encoder_weights.pt")
        )
        torch.save(
            self.decoder.state_dict(), os.path.join(output_path, "decoder_weights.pt")
        )
        return output_path

    def load_weights(self, input_path):
        self.encoder.load_state_dict(
            torch.load(os.path.join(input_path, "encoder_weights.pt"))
        )
        self.decoder.load_state_dict(
            torch.load(os.path.join(input_path, "decoder_weights.pt"))
        )
        return input_path
