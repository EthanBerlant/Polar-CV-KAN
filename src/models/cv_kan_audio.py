"""
CV-KAN Audio/Speech Processing Model.

A CV-KAN variant optimized for audio processing that leverages the 
natural complex structure of Short-Time Fourier Transform (STFT) 
spectrograms.

Key insight: STFT output is already complex-valued, so CV-KAN can
directly process frequency-domain audio without artificial conversion.
Frequency bins become complex dimensions, time frames become tokens.

Supports:
- Direct complex STFT input (no embedding needed)
- Audio classification (speech commands, music genre, etc.)
- Audio reconstruction via inverse STFT
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

from ..modules.polarizing_block import PolarizingBlock
from ..modules.aggregation import GlobalMeanAggregation
from ..modules.positional_encoding import ComplexPositionalEncoding


class STFTFrontend(nn.Module):
    """
    STFT frontend that converts waveform to complex spectrogram.
    
    Optional module for when raw waveforms are provided instead
    of pre-computed spectrograms.
    
    Args:
        n_fft: FFT window size
        hop_length: Hop between windows
        win_length: Window length (default: n_fft)
        window: Window function ('hann', 'hamming', 'blackman')
        center: Whether to pad signal at beginning and end
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        
        # Register window as buffer
        if window == 'hann':
            win = torch.hann_window(self.win_length)
        elif window == 'hamming':
            win = torch.hamming_window(self.win_length)
        elif window == 'blackman':
            win = torch.blackman_window(self.win_length)
        else:
            raise ValueError(f"Unknown window: {window}")
        
        self.register_buffer('window', win)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT of waveform.
        
        Args:
            waveform: Audio tensor (batch, samples) or (batch, channels, samples)
        
        Returns:
            Complex spectrogram (batch, time_frames, n_freq_bins)
        """
        # Handle stereo by averaging to mono
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )  # (batch, n_freq, time)
        
        # Transpose to (batch, time, freq) for CV-KAN
        stft = stft.transpose(1, 2)
        
        return stft
    
    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1


class ISTFTBackend(nn.Module):
    """
    Inverse STFT backend for audio reconstruction.
    
    Args:
        n_fft: FFT window size (must match frontend)
        hop_length: Hop between windows
        win_length: Window length
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
        self.register_buffer('window', torch.hann_window(self.win_length))
    
    def forward(
        self, 
        spectrogram: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reconstruct waveform from complex spectrogram.
        
        Args:
            spectrogram: Complex tensor (batch, time, freq)
            length: Target output length (optional)
        
        Returns:
            Waveform (batch, samples)
        """
        # Transpose back to (batch, freq, time)
        stft = spectrogram.transpose(1, 2)
        
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        
        return waveform


class FrequencyProjection(nn.Module):
    """
    Project frequency bins to target complex dimension.
    
    Used when n_freq_bins != d_complex.
    
    Args:
        n_freq: Number of frequency bins
        d_complex: Target complex dimension
    """
    
    def __init__(self, n_freq: int, d_complex: int):
        super().__init__()
        self.needs_projection = n_freq != d_complex
        
        if self.needs_projection:
            # Complex linear projection
            self.weight = nn.Parameter(
                torch.randn(n_freq, d_complex, dtype=torch.cfloat) * 0.02
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_projection:
            return torch.einsum('...i,io->...o', x, self.weight)
        return x
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Project back to frequency domain."""
        if self.needs_projection:
            # Pseudo-inverse projection
            return torch.einsum('...o,io->...i', x, self.weight.conj())
        return x


class CVKANAudio(nn.Module):
    """
    CV-KAN model for audio/speech processing.
    
    Leverages the natural complex structure of STFT spectrograms.
    Frequency bins are treated as complex dimensions, time frames
    as tokens (sequence positions).
    
    Args:
        n_fft: FFT size for STFT (determines frequency resolution)
        hop_length: STFT hop length
        d_complex: Internal complex dimension (None = use n_freq_bins)
        n_layers: Number of CV-KAN layers
        n_classes: Number of classes for classification (None for reconstruction)
        kan_hidden: Hidden size for KAN MLPs
        task: 'classification' or 'reconstruction'
        pooling: Pooling strategy for classification ('mean', 'max', 'attention')
        use_stft_frontend: Whether to compute STFT (False if input is spectrogram)
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        d_complex: Optional[int] = None,
        n_layers: int = 4,
        n_classes: Optional[int] = None,
        kan_hidden: int = 32,
        task: Literal['classification', 'reconstruction'] = 'classification',
        pooling: Literal['mean', 'max', 'attention'] = 'mean',
        use_stft_frontend: bool = True,
    ):
        super().__init__()
        self.task = task
        self.pooling = pooling
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        n_freq_bins = n_fft // 2 + 1
        d_complex = d_complex or n_freq_bins
        self.d_complex = d_complex
        self.n_freq_bins = n_freq_bins
        
        # STFT frontend (optional)
        if use_stft_frontend:
            self.stft_frontend = STFTFrontend(
                n_fft=n_fft,
                hop_length=hop_length,
            )
        else:
            self.stft_frontend = None
        
        # Frequency projection (if dimensions don't match)
        self.freq_proj = FrequencyProjection(n_freq_bins, d_complex)
        
        # Positional encoding for time dimension
        self.pos_encoding = ComplexPositionalEncoding(d_complex)
        
        # CV-KAN layers
        self.layers = nn.ModuleList([
            PolarizingBlock(d_complex, kan_hidden)
            for _ in range(n_layers)
        ])
        
        # Task-specific heads
        if task == 'classification':
            assert n_classes is not None, "n_classes required for classification"
            
            if pooling == 'attention':
                # Learnable query for attention pooling
                self.pool_query = nn.Parameter(
                    torch.randn(1, 1, d_complex, dtype=torch.cfloat) * 0.02
                )
            
            self.classifier = nn.Sequential(
                nn.Linear(d_complex, kan_hidden * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(kan_hidden * 2, n_classes),
            )
        
        elif task == 'reconstruction':
            # Inverse frequency projection
            self.freq_proj_inv = FrequencyProjection(d_complex, n_freq_bins)
            
            # Optional: learnable reconstruction refinement
            self.refine = nn.Sequential(
                nn.Linear(n_freq_bins * 2, kan_hidden),
                nn.GELU(),
                nn.Linear(kan_hidden, n_freq_bins * 2),
            )
            
            # ISTFT backend
            self.istft_backend = ISTFTBackend(
                n_fft=n_fft,
                hop_length=hop_length,
            )
    
    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pool across time dimension.
        
        Args:
            z: Complex tensor (batch, time, d_complex)
        
        Returns:
            Pooled tensor (batch, d_complex)
        """
        if self.pooling == 'mean':
            return z.mean(dim=1)
        elif self.pooling == 'max':
            mags = torch.abs(z)
            max_idx = mags.sum(dim=-1).argmax(dim=1, keepdim=True)
            return torch.gather(
                z, 1, 
                max_idx.unsqueeze(-1).expand(-1, -1, self.d_complex)
            ).squeeze(1)
        elif self.pooling == 'attention':
            # Attention-based pooling using phase similarity
            query = self.pool_query.expand(z.shape[0], -1, -1)
            # Compute attention via complex inner product
            attn = torch.einsum('btd,bqd->btq', z, query.conj())
            attn = torch.softmax(attn.abs(), dim=1)
            return torch.einsum('btq,btd->bqd', attn, z).squeeze(1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_spectrogram: bool = False,
        output_length: Optional[int] = None,
    ) -> dict:
        """
        Forward pass for audio processing.
        
        Args:
            x: Input tensor:
               - Waveform (batch, samples) if use_stft_frontend=True
               - Complex spectrogram (batch, time, freq) otherwise
            return_spectrogram: Whether to return processed spectrogram
            output_length: Target waveform length for reconstruction
        
        Returns:
            Dictionary with task-specific outputs:
            - Classification: 'logits', optionally 'features'
            - Reconstruction: 'waveform', optionally 'spectrogram'
        """
        # Compute STFT if needed
        if self.stft_frontend is not None:
            z = self.stft_frontend(x)  # (batch, time, n_freq)
        else:
            z = x  # Already complex spectrogram
        
        original_spec = z if return_spectrogram else None
        
        # Project to working dimension
        z = self.freq_proj(z)
        
        # Apply positional encoding
        z = self.pos_encoding(z)
        
        # CV-KAN layers
        for layer in self.layers:
            z = layer(z)
        
        output = {}
        
        if self.task == 'classification':
            # Pool and classify
            pooled = self._pool(z)
            features = torch.abs(pooled)
            logits = self.classifier(features)
            
            output['logits'] = logits
            output['features'] = features
            
        elif self.task == 'reconstruction':
            # Project back to frequency domain
            z_freq = self.freq_proj_inv.inverse(z)
            
            # Refine (operating on real/imag concatenation)
            z_cat = torch.cat([z_freq.real, z_freq.imag], dim=-1)
            z_refined = z_cat + self.refine(z_cat)
            z_freq = torch.complex(
                z_refined[..., :self.n_freq_bins],
                z_refined[..., self.n_freq_bins:]
            )
            
            # Reconstruct waveform
            waveform = self.istft_backend(z_freq, length=output_length)
            
            output['waveform'] = waveform
            output['spectrogram'] = z_freq
        
        if return_spectrogram and original_spec is not None:
            output['input_spectrogram'] = original_spec
        
        return output
    
    @classmethod
    def for_speech_commands(
        cls,
        n_classes: int = 35,
        sample_rate: int = 16000,
        **kwargs,
    ) -> 'CVKANAudio':
        """
        Factory for speech command classification.
        
        Preconfigured for typical speech command datasets (1-second clips).
        """
        return cls(
            n_fft=512,
            hop_length=128,
            d_complex=128,
            n_layers=4,
            n_classes=n_classes,
            task='classification',
            pooling='mean',
            **kwargs,
        )
    
    @classmethod
    def for_music_tagging(
        cls,
        n_classes: int = 50,
        **kwargs,
    ) -> 'CVKANAudio':
        """
        Factory for music tagging/genre classification.
        
        Preconfigured for longer audio clips with higher frequency resolution.
        """
        return cls(
            n_fft=2048,
            hop_length=512,
            d_complex=256,
            n_layers=6,
            n_classes=n_classes,
            task='classification',
            pooling='attention',
            **kwargs,
        )
