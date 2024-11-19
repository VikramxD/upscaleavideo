from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Tuple, Union
from loguru import logger
import sys
import os
import subprocess
import time
import warnings
import torch
import math
import imageio
import textwrap
import numpy as np
import cv2
from PIL import Image
from einops import rearrange
from torch.nn import functional as F
import transformers

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
transformers.logging.set_verbosity_error()

# Import model components
from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import wavelet_reconstruction, adaptive_instance_normalization
from llava.llava_agent import LLavaAgent

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    "logs/upscale_video_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="500 MB",
    retention="10 days"
)

# Constants
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

class UpscaleConfig(BaseModel):
    """Configuration for Upscale-A-Video inference pipeline."""
    
    # Model paths and settings
    model_cache: Path = Field(default=Path("model_cache"))
    model_url: str = Field(
        default="https://weights.replicate.delivery/default/sczhou/Upscale-A-Video/model_cache.tar"
    )
    
    # I/O settings
    input_path: Path = Field(default=Path("inputs"))
    output_path: Path = Field(default=Path("results"))
    
    # Model parameters
    use_video_vae: bool = Field(default=False)
    noise_level: int = Field(default=150, ge=0, le=200)
    guidance_scale: float = Field(default=6.0, ge=0, le=20)
    inference_steps: int = Field(default=30, ge=0, le=100)
    propagation_steps: List[int] = Field(default_factory=list)
    color_fix: Literal["None", "AdaIn", "Wavelet"] = Field(default="None")
    
    # Generation settings
    a_prompt: str = Field(default="best quality, extremely detailed")
    n_prompt: str = Field(default="blur, worst quality")
    use_llava: bool = Field(default=False)
    load_8bit_llava: bool = Field(default=True)
    seed: Optional[int] = Field(default=None)
    tile_size: int = Field(default=256)
    
    def setup_environment(self):
        """Configure environment for offline model usage."""
        env_vars = {
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(self.model_cache),
            "TORCH_HOME": str(self.model_cache),
            "HF_DATASETS_CACHE": str(self.model_cache),
            "TRANSFORMERS_CACHE": str(self.model_cache),
            "HUGGINGFACE_HUB_CACHE": str(self.model_cache)
        }
        for key, value in env_vars.items():
            os.environ[key] = value

def verify_downloaded_files(model_cache: Path) -> bool:
    """Verify if required model files exist in cache."""
    required_files = [
        "upscale_a_video/vae/vae_3d.bin",
        "upscale_a_video/vae/vae_video.bin",
        "upscale_a_video/unet/unet_video.bin",
        "upscale_a_video/propagator/raft-things.pth"
    ]
    return all((model_cache / file).exists() for file in required_files)

def download_weights(config: UpscaleConfig) -> None:
    """Download and extract model weights."""
    if verify_downloaded_files(config.model_cache):
        logger.info("Required model files already exist in cache")
        return
        
    config.model_cache.mkdir(parents=True, exist_ok=True)
    tar_path = config.model_cache / "model_cache.tar"
    
    try:
        # Download with wget showing progress
        logger.info(f"Downloading weights from: {config.model_url}")
        subprocess.check_call([
            "wget",
            "--progress=bar:force",
            "--show-progress",
            "-O", str(tar_path),
            config.model_url
        ], stderr=subprocess.STDOUT)
        
        # Extract tar file
        logger.info("Extracting weights...")
        subprocess.check_call([
            "tar",
            "-xf",
            str(tar_path),
            "-C", str(config.model_cache),
            "--strip-components=1"
        ])
        
        # Verify extraction
        if not verify_downloaded_files(config.model_cache):
            raise RuntimeError("Model files missing after extraction")
            
        # Cleanup
        tar_path.unlink()
        logger.info("Download and extraction complete")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        if tar_path.exists():
            tar_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Failed to process weights: {str(e)}")
        if tar_path.exists():
            tar_path.unlink()
        raise

def setup_devices() -> Tuple[str, str]:
    """Configure CUDA devices for inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
        
    if torch.cuda.device_count() >= 2:
        return 'cuda:0', 'cuda:1'
    return 'cuda:0', 'cuda:0'

def read_video(video_path: Union[str, Path]) -> Tuple[torch.Tensor, float, tuple, str]:
    """Read video frames and return as tensor."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
        
    frames = np.stack(frames)
    frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))
    size = frames.shape[2:]
    video_name = Path(video_path).stem
    
    return frames, fps, size, video_name

class UpscaleVideoInference:
    def __init__(self, config: UpscaleConfig):
        """Initialize upscaling pipeline."""
        self.config = config
        logger.info("Initializing Upscale-A-Video pipeline")
        
        # Setup environment and devices
        self.config.setup_environment()
        download_weights(self.config)
        self.uav_device, self.llava_device = setup_devices()
        
        # Load models
        self._load_models()
        
    def _load_models(self) -> None:
        """Load all required models."""
        try:
            logger.info("Loading models")
            
            # Load pipeline
            self.pipeline = VideoUpscalePipeline.from_pretrained(
                str(self.config.model_cache / "upscale_a_video"),
                torch_dtype=torch.float16
            )
            
            # Load VAE
            self._load_vae()
            
            # Load UNet
            self._load_unet()
            
            # Load scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(
                str(self.config.model_cache / "upscale_a_video/scheduler/scheduler_config.json")
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.uav_device)
            
            # Load LLaVA if needed
            if self.config.use_llava:
                self._load_llava()
                
            # Load propagator if needed
            if self.config.propagation_steps:
                self._load_propagator()
                
            logger.success("All models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load models")
            raise RuntimeError(f"Model loading failed: {str(e)}")
            
    def _load_vae(self) -> None:
        """Load VAE model."""
        config_path = "vae_video_config.json" if self.config.use_video_vae else "vae_3d_config.json"
        weights_path = "vae_video.bin" if self.config.use_video_vae else "vae_3d.bin"
        
        self.pipeline.vae = AutoencoderKLVideo.from_config(
            str(self.config.model_cache / f"upscale_a_video/vae/{config_path}")
        )
        self.pipeline.vae.load_state_dict(
            torch.load(
                str(self.config.model_cache / f"upscale_a_video/vae/{weights_path}"),
                map_location="cpu"
            )
        )
        
    def _load_unet(self) -> None:
        """Load UNet model."""
        self.pipeline.unet = UNetVideoModel.from_config(
            str(self.config.model_cache / "upscale_a_video/unet/unet_video_config.json")
        )
        self.pipeline.unet.load_state_dict(
            torch.load(
                str(self.config.model_cache / "upscale_a_video/unet/unet_video.bin"),
                map_location="cpu"
            ),
            strict=True
        )
        self.pipeline.unet = self.pipeline.unet.half()
        self.pipeline.unet.eval()
        
    def _load_llava(self) -> None:
        """Load LLaVA model."""
        self.llava_agent = LLavaAgent(
            str(self.config.model_cache / "liuhaotian-llava-v1.5-13b"),
            device=self.llava_device,
            load_8bit=self.config.load_8bit_llava
        )
        
    def _load_propagator(self) -> None:
        """Load RAFT and propagator."""
        self.raft = RAFT_bi(
            str(self.config.model_cache / "upscale_a_video/propagator/raft-things.pth")
        )
        self.propagator = Propagation(4, learnable=False)
        self.pipeline.propagator = self.propagator

    def _generate_caption(self, image: torch.Tensor) -> str:
        """Generate caption using LLaVA."""
        if not self.config.use_llava:
            return ""
            
        logger.info("Generating caption with LLaVA")
        
        with torch.no_grad():
            # Prepare image for LLaVA
            w, h = image.shape[-1], image.shape[-2]
            fix_resize = 512
            _scale = fix_resize / min(w, h)
            w0, h0 = round(w * _scale), round(h * _scale)
            
            image = F.interpolate(
                image.unsqueeze(0).float(),
                size=(h0, w0),
                mode="bicubic"
            )
            image = (image.squeeze(0).permute(1, 2, 0)).cpu().numpy()
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
            
            caption = self.llava_agent.gen_image_caption([image])[0]
            
        logger.info(f"Generated caption: {caption}")
        return caption

    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Preprocess video frames."""
        frames = (frames/255. - 0.5) * 2  # Normalize to [-1, 1]
        frames = frames.to(self.uav_device)

        h, w = frames.shape[-2:]
        if h >= 1280 and w >= 1280:
            frames = F.interpolate(frames, (int(h//4), int(w//4)), mode="area")

        frames = frames.unsqueeze(0)  # Add batch dimension
        frames = rearrange(frames, 'b t c h w -> b c t h w')
        
        return frames.contiguous()

    def process_video(self, video_path: Path) -> Path:
        """Process a single video through the pipeline."""
        logger.info(f"Processing video: {video_path}")
        
        # Load video
        vframes, fps, size, video_name = read_video(video_path)
        
        # Generate caption if needed
        prompt = self._generate_caption(vframes[0]) + self.config.a_prompt \
            if self.config.use_llava else self.config.a_prompt
            
        logger.debug(f"Using prompt: {prompt}")
            
        # Preprocess frames
        vframes = self._preprocess_frames(vframes)
        
        # Setup flows if needed
        if self.config.propagation_steps:
            flows_forward, flows_backward = self.raft.forward_slicing(vframes)
            flows_bi = [flows_forward, flows_backward]
        else:
            flows_bi = None
        
        # Process frames
        torch.cuda.synchronize()
        start_time = time.time()
        
        if self.config.tile_size and vframes.shape[-1] * vframes.shape[-2] >= 384*384:
            output = self._process_with_tiling(vframes, prompt, flows_bi)
        else:
            output = self._process_without_tiling(vframes, prompt, flows_bi)
            
        # Apply color correction
        output = self._apply_color_correction(output, vframes)
        
        torch.cuda.synchronize()
        process_time = time.time() - start_time
        
        # Save results
        output_path = self._save_results(output, video_name, fps)
        logger.info(f"Processing completed in {process_time:.2f}s")
        
        return output_path

    def _process_with_tiling(
        self,
        vframes: torch.Tensor,
        prompt: str,
        flows_bi: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Process video using tiling strategy."""
        b, c, t, h, w = vframes.shape
        tile_size = self.config.tile_size
        overlap = 64
        
        output_h = h * 4
        output_w = w * 4
        output = vframes.new_zeros((b, c, t, output_h, output_w))
        
        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)
        
        logger.info(f"Processing with tiling: {tiles_x}x{tiles_y} tiles")
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                logger.debug(f"Processing tile [{y+1}/{tiles_y}] x [{x+1}/{tiles_x}]")
                
                # Calculate tile coordinates
                start_x = x * tile_size
                end_x = min((x + 1) * tile_size, w)
                start_y = y * tile_size
                end_y = min((y + 1) * tile_size, h)
                
                # Add overlap
                pad_start_x = max(start_x - overlap, 0)
                pad_end_x = min(end_x + overlap, w)
                pad_start_y = max(start_y - overlap, 0)
                pad_end_y = min(end_y + overlap, h)
                
                # Extract tile
                tile = vframes[..., pad_start_y:pad_end_y, pad_start_x:pad_end_x]
                
                # Process flows for tile if needed
                if flows_bi is not None:
                    flows_tile = [
                        flows_bi[0][..., pad_start_y:pad_end_y, pad_start_x:pad_end_x],
                        flows_bi[1][..., pad_start_y:pad_end_y, pad_start_x:pad_end_x]
                    ]
                else:
                    flows_tile = None
                
                # Process tile
                try:
                    with torch.no_grad():
                        output_tile = self.pipeline(
                            prompt,
                            image=tile,
                            flows_bi=flows_tile,
                            generator=torch.Generator(device=self.uav_device).manual_seed(
                                self.config.seed or int.from_bytes(os.urandom(2), "big")
                            ),
                            num_inference_steps=self.config.inference_steps,
                            guidance_scale=self.config.guidance_scale,
                            noise_level=self.config.noise_level,
                            negative_prompt=self.config.n_prompt,
                            propagation_steps=self.config.propagation_steps,
                        ).images
                except RuntimeError as e:
                    logger.error(f"Error processing tile: {e}")
                    raise
                
                # Calculate output coordinates
                out_start_x = start_x * 4
                out_end_x = end_x * 4
                out_start_y = start_y * 4
                out_end_y = end_y * 4
                
                # Calculate tile output coordinates
                tile_start_x = (start_x - pad_start_x) * 4
                tile_end_x = tile_start_x + (end_x - start_x) * 4
                tile_start_y = (start_y - pad_start_y) * 4
                tile_end_y = tile_start_y + (end_y - start_y) * 4
                
                # Place tile in output
                output[..., out_start_y:out_end_y, out_start_x:out_end_x] = \
                    output_tile[..., tile_start_y:tile_end_y, tile_start_x:tile_end_x]
                
        return output

    def _process_without_tiling(
        self,
        vframes: torch.Tensor,
        prompt: str,
        flows_bi: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Process video without tiling."""
        logger.info("Processing without tiling")
        
        try:
            with torch.no_grad():
                output = self.pipeline(
                    prompt,
                    image=vframes,
                    flows_bi=flows_bi,
                    generator=torch.Generator(device=self.uav_device).manual_seed(
                        self.config.seed or int.from_bytes(os.urandom(2), "big")
                    ),
                    num_inference_steps=self.config.inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    noise_level=self.config.noise_level,
                    negative_prompt=self.config.n_prompt,
                    propagation_steps=self.config.propagation_steps,
                ).images
        except RuntimeError as e:
            logger.error(f"Error processing video: {e}")
            raise
            
        return output

    def _apply_color_correction(
        self,
        output: torch.Tensor,
        vframes: torch.Tensor
    ) -> torch.Tensor:
        """Apply color correction to processed frames."""
        if self.config.color_fix in ["AdaIn", "Wavelet"]:
            logger.info(f"Applying {self.config.color_fix} color correction")
            
            vframes = rearrange(vframes.squeeze(0), 'c t h w -> t c h w')
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w')
            vframes = F.interpolate(vframes, scale_factor=4, mode="bicubic")
            
            if self.config.color_fix == "AdaIn":
                output = adaptive_instance_normalization(output, vframes)
            else:  # Wavelet
                output = wavelet_reconstruction(output, vframes)
        else:
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w')
            
        return output

    def _save_results(
        self,
        output: torch.Tensor,
        video_name: str,
        fps: float
    ) -> Path:
        """Save processed video."""
        # Generate output filename
        suffix = ""
        if self.config.propagation_steps:
            suffix += f"_p{'_'.join(map(str, self.config.propagation_steps))}"
        
        save_name = (
            f"{video_name}_n{self.config.noise_level}"
            f"_g{self.config.guidance_scale}"
            f"_s{self.config.inference_steps}{suffix}"
        )
        
        # Prepare output directory
        save_dir = self.config.output_path / "video"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{save_name}.mp4"
        
        # Prepare video frames
        output = output.cpu()
        output = (output / 2 + 0.5).clamp(0, 1) * 255
        output = rearrange(output, 't c h w -> t h w c')
        output = output.cpu().numpy().astype(np.uint8)
        
        # Save video with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                imageio.mimwrite(
                    str(save_path),
                    output,
                    fps=fps,
                    quality=8,
                    output_params=["-loglevel", "error"]
                )
                logger.info(f"Saved video to: {save_path}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to save video after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            
        return save_path

def main():
    """Main entry point for video upscaling."""
    try:
        # Initialize config and pipeline
        config = UpscaleConfig()
        
        # Log system info
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
        
        # Initialize upscaler
        upscaler = UpscaleVideoInference(config)
        
        # Get video paths
        input_path = config.input_path
        if input_path.is_file():
            if not str(input_path).endswith(VIDEO_EXTENSIONS):
                raise ValueError(f"Video format must be one of {VIDEO_EXTENSIONS}")
            video_paths = [input_path]
        else:
            video_paths = [p for p in input_path.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
            
        total_videos = len(video_paths)
        if total_videos == 0:
            logger.warning(f"No video files found in {input_path}")
            return
            
        logger.info(f"Found {total_videos} videos to process")
        
        # Process videos
        successful = 0
        failed = 0
        
        for idx, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing video {idx}/{total_videos}: {video_path}")
            try:
                # Check file size
                file_size = video_path.stat().st_size / (1024 * 1024)  # Size in MB
                logger.info(f"Video size: {file_size:.2f}MB")
                
                output_path = upscaler.process_video(video_path)
                successful += 1
                logger.success(f"Successfully processed: {video_path} -> {output_path}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {video_path}: {str(e)}")
                continue
                
        # Log summary
        logger.info("\nProcessing Summary:")
        logger.info("=" * 50)
        logger.info(f"Total videos: {total_videos}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        if failed > 0:
            logger.warning("Some videos failed to process. Check logs for details.")
                
    except Exception as e:
        logger.exception("Fatal error")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"Starting Upscale-A-Video at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    main()