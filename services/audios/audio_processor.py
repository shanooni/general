import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from pathlib import Path
import warnings
import io
warnings.filterwarnings('ignore')

# Try to import audio libraries with fallbacks
AUDIO_BACKEND = None
try:
    import soundfile as sf
    AUDIO_BACKEND = "soundfile"
except ImportError:
    try:
        import librosa
        AUDIO_BACKEND = "librosa"
    except ImportError:
        try:
            import torchaudio
            torchaudio.set_audio_backend("sox_io")
            AUDIO_BACKEND = "torchaudio"
        except:
            pass

if AUDIO_BACKEND:
    print(f"Using audio backend: {AUDIO_BACKEND}")
else:
    print("Warning: No audio backend found. Install soundfile, librosa, or torchaudio")


class MultiLayerAudioFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", layer_indices=None):
        """
        Initialize Wav2Vec2 multi-layer feature extractor for REST API usage
        
        Args:
            model_name: HuggingFace model identifier
            layer_indices: List of layer indices to extract from (default: [3, 6, 9, 12])
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.layer_indices = layer_indices if layer_indices else [3, 6, 9, 12]
        print(f"Model loaded on {self.device}")
        print(f"Extracting from layers: {self.layer_indices}")
    
    def extract_features_from_file(self, audio_file):
        """
        Extract multi-layer features from an audio file (for REST endpoint)
        
        Args:
            audio_file: Can be:
                - File path (str or Path)
                - File-like object (BytesIO, UploadFile, etc.)
                - Bytes data
        
        Returns:
            numpy array of multi-layer features
        """
        waveform = None
        sample_rate = None
        
        # Prepare file input
        if isinstance(audio_file, bytes):
            audio_file = io.BytesIO(audio_file)
        
        # Try different loading methods with proper error handling
        try:
            if AUDIO_BACKEND == "soundfile":
                import soundfile as sf
                if isinstance(audio_file, (str, Path)):
                    data, sample_rate = sf.read(str(audio_file))
                else:
                    data, sample_rate = sf.read(audio_file)
                waveform = torch.from_numpy(data).float()
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                    
            elif AUDIO_BACKEND == "librosa":
                import librosa
                if isinstance(audio_file, (str, Path)):
                    data, sample_rate = librosa.load(str(audio_file), sr=None)
                else:
                    # Save bytes to temp file for librosa
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    data, sample_rate = librosa.load(tmp_path, sr=None)
                    Path(tmp_path).unlink()  # Clean up temp file
                    
                waveform = torch.from_numpy(data).float().unsqueeze(0)
                
            elif AUDIO_BACKEND == "torchaudio":
                import torchaudio
                if isinstance(audio_file, (str, Path)):
                    waveform, sample_rate = torchaudio.load(str(audio_file))
                else:
                    waveform, sample_rate = torchaudio.load(audio_file)
            else:
                raise RuntimeError(
                    "No audio backend available. Please install one of:\n"
                    "  pip install soundfile\n"
                    "  pip install librosa\n"
                    "  pip install torchaudio"
                )
                
        except Exception as e:
            raise ValueError(f"Failed to load audio file using {AUDIO_BACKEND}: {str(e)}")
        
        if waveform is None:
            raise ValueError("Failed to load audio waveform")
        
        # Resample to 16kHz if needed (wav2vec2 expects 16kHz)
        if sample_rate != 16000:
            if AUDIO_BACKEND == "librosa":
                import librosa
                data_resampled = librosa.resample(
                    waveform.squeeze().numpy(), 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                waveform = torch.from_numpy(data_resampled).unsqueeze(0)
            else:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process with wav2vec2
        inputs = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract multi-layer features
            features = self._extract_multi_layer(outputs.hidden_states)
            
        return features
    
    def _extract_multi_layer(self, hidden_states):
        """Extract and concatenate features from multiple transformer layers"""
        layer_features = []
        for idx in self.layer_indices:
            if idx < len(hidden_states):
                layer_feat = hidden_states[idx].squeeze(0).cpu().numpy()
                # Aggregate each layer with mean pooling
                layer_features.append(np.mean(layer_feat, axis=0))
        return np.concatenate(layer_features)
    
    def get_feature_dimension(self):
        """Get the output feature dimension"""
        # Base wav2vec2 has 768 dimensions per layer
        return len(self.layer_indices) * 768
