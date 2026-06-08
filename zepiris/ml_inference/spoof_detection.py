"""Anti-spoofing / liveness detection service using MobileNetV3-Large."""

from __future__ import annotations

import io

import cv2
import numpy as np
import torch

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.ml_inference.models import MobileNetV3LSpoof
from zepiris.ml_inference.moire_detection import ScreenReplayDetector
from zepiris.schemas.ml_inference import SpoofDetectionResult

INPUT_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SpoofDetectionService(ModelService):
    """Anti-spoofing service using MobileNetV3-Large binary classifier.

    Architecture: MobileNetV3-Large with 1-output linear head.
    Input: RGB image resized to 224x224, ImageNet-normalized.
    Output: sigmoid of logit gives prob_live; > 0.5 means genuine.
    Model weights (checkpoint with "model_state_dict" key) downloaded from Hugging Face Hub.
    """

    def __init__(
        self,
        huggingface_repo_id: str,
        huggingface_model_file: str = "model.pth",
        local_model_path: str | None = None,
        model_source: str = "auto",
        spoof_threshold: float = 0.5,
        screen_replay_detector: ScreenReplayDetector | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize anti-spoofing service.

        Args:
            huggingface_repo_id: Hugging Face repository ID (e.g., "user/spoof-detection")
            huggingface_model_file: Filename of model weights in the repo
            local_model_path: Optional local path to model weights
            model_source: ``"auto"``, ``"local"``, or ``"huggingface"``
            spoof_threshold: Threshold for the learned-model liveness probability
                (``prob_live`` must exceed this; default 0.5).
            screen_replay_detector: Optional passive moiré/glare detector. When
                provided, an image is only ``is_live`` if the model passes AND
                the detector does not flag a recaptured screen.
            device: Inference device ("cpu" or "cuda")
        """
        self._spoof_threshold = spoof_threshold
        self._screen_replay_detector = screen_replay_detector
        self._model: MobileNetV3LSpoof | None = None
        config = ModelServiceConfig(
            model_name="spoof_detection",
            huggingface_repo_id=huggingface_repo_id,
            huggingface_model_file=huggingface_model_file,
            local_model_path=local_model_path,
            model_source=model_source,
            device=device,
        )
        super().__init__(config)

    def load_model(self) -> MobileNetV3LSpoof:
        """Load checkpoint from configured source and apply model_state_dict.

        Returns:
            MobileNetV3LSpoof: Model in eval mode on configured device
        """
        if self._model is not None:
            return self._model

        model_bytes = self._download_model()
        buffer = io.BytesIO(model_bytes)

        # Full training checkpoint (dict with model_state_dict); not weights-only safe.
        checkpoint = torch.load(buffer, map_location=self.config.device, weights_only=False)
        model = MobileNetV3LSpoof()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.config.device)
        model.eval()

        self._model = model
        return self._model

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Resize to 224x224, ImageNet-normalize, convert to NCHW tensor.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: {"tensor": torch.Tensor of shape (1, 3, 224, 224)}
        """
        resized = cv2.resize(image_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)
        return {"tensor": tensor}

    def predict(self, preprocessed_data: dict) -> np.ndarray:
        """Run spoof model forward pass and apply sigmoid.

        Args:
            preprocessed_data: dict with "tensor" key from preprocess()

        Returns:
            np.ndarray: [prob_live, prob_spoof], shape (2,)
        """
        model = self.load_model()
        tensor = preprocessed_data["tensor"].to(self.config.device)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob_live = torch.sigmoid(logit).item()
        prob_spoof = 1.0 - prob_live
        return np.array([prob_live, prob_spoof], dtype=np.float32)

    def postprocess(self, output: np.ndarray) -> SpoofDetectionResult:
        """Convert live/spoof probabilities to SpoofDetectionResult.

        Args:
            output: np.ndarray [prob_live, prob_spoof], shape (2,)

        Returns:
            SpoofDetectionResult: is_live=True when prob_live > threshold
        """
        prob_live, _ = float(output[0]), float(output[1])
        is_live = prob_live > self._spoof_threshold

        return SpoofDetectionResult(
            is_live=is_live,
            probability=prob_live,
        )

    def forward(self, image_rgb: np.ndarray) -> SpoofDetectionResult:
        """Run the learned spoof model, then AND-gate with screen-replay detection.

        The image is declared live only if the model's ``prob_live`` clears the
        threshold AND the passive moiré/glare detector (when configured) does
        not flag a recaptured screen. Either signal alone can mark it spoofed.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            SpoofDetectionResult: ``is_live`` reflects the fused decision;
                ``probability`` remains the model's live probability.
        """
        result = super().forward(image_rgb)
        assert isinstance(result, SpoofDetectionResult)  # noqa: S101

        if self._screen_replay_detector is None:
            return result

        screen = self._screen_replay_detector.analyze(image_rgb)
        if screen.is_screen:
            return SpoofDetectionResult(is_live=False, probability=result.probability)
        return result
