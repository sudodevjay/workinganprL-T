"""Runs the SAR (attention) head from the same indian_plate_rec_v2 checkpoint
already deployed for CTC, to get a second, more accurate reading of whichever
single crop CTC's fast multi-variant search already picked as the best guess.

Why this exists: eval_ocr_accuracy.py-style comparison across all 253
held-out images showed raw SAR at 98.81% exact-match vs the full deployed
CTC+grammar-correction pipeline at 87.75% -- SAR doesn't have CTC's
blank-collapse failure mode (the "drops/duplicates a character" problem), so
every one of that comparison's disagreements was CTC-wrong/SAR-right, never
the other way round. But SAR's autoregressive decode is ~6-7x slower per call
than CTC in local benchmarks, so it's used only ONCE per vehicle -- as a
refinement of the already-cheap CTC variant-search's winning crop -- not to
replace that search.

Needs the `ppocr` package vendored at vendor/ppocr (just ppocr.modeling +
ppocr.postprocess, not the full PaddleOCR repo) plus scikit-image/shapely/
pyclipper installed in this venv -- see the SAR integration notes for how
those were added.
"""
import math
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "vendor"))

import paddle  # noqa: E402  (after sys.path insert)
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402

IMG_H = 48
IMG_W = 320

# Mirrors ocr_training/release/train_config.yml's Architecture block as it
# was when weights/indian_plate_rec_v2_sar_best_accuracy.pdparams was
# trained (backbone scale 0.5). That yaml has since been bumped to 0.75 to
# prepare the *next* retrain -- do not point this at that file; a mismatched
# scale silently mis-loads (or fails to load) weights. Re-derive this from
# whatever config actually produced the checkpoint currently in use.
ARCHITECTURE_CONFIG = {
    "model_type": "rec",
    "algorithm": "SVTR_LCNet",
    "Transform": None,
    "Backbone": {
        "name": "MobileNetV1Enhance",
        "scale": 0.5,
        "last_conv_stride": [1, 2],
        "last_pool_type": "avg",
        "last_pool_kernel_size": [2, 2],
    },
    "Head": {
        "name": "MultiHead",
        "head_list": [
            {
                "CTCHead": {
                    "Neck": {"name": "svtr", "dims": 64, "depth": 2, "hidden_dims": 120, "use_guide": True},
                    "Head": {"fc_decay": 0.00001},
                }
            },
            {"SARHead": {"enc_dim": 512, "max_text_length": 25}},
        ],
    },
}


class SARRefiner:
    def __init__(self, checkpoint_path: str, char_dict_path: str, use_space_char: bool = True):
        global_config = {
            "character_dict_path": char_dict_path,
            "use_space_char": use_space_char,
            "max_text_length": 25,
        }
        ctc_post = build_post_process({"name": "CTCLabelDecode"}, global_config)
        char_num = len(ctc_post.character)
        arch_config = dict(ARCHITECTURE_CONFIG)
        arch_config["Head"] = dict(arch_config["Head"])
        arch_config["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
        }

        self.model = build_model(arch_config)
        state_dict = paddle.load(checkpoint_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()

        self.sar_post = build_post_process({"name": "SARLabelDecode"}, global_config)

    def _resize_norm(self, img):
        h, w = img.shape[:2]
        ratio = w / float(h)
        max_wh_ratio = max(IMG_W / float(IMG_H), ratio)
        img_w = int(IMG_H * max_wh_ratio)
        resized_w = img_w if math.ceil(IMG_H * ratio) > img_w else math.ceil(IMG_H * ratio)

        resized = cv2.resize(img, (resized_w, IMG_H))
        resized = resized.astype("float32").transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5

        padded = np.zeros((3, IMG_H, img_w), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        valid_ratio = min(1.0, resized_w / img_w)
        return padded, valid_ratio

    def recognize(self, img):
        if img is None or img.size == 0:
            return "", 0.0

        padded, valid_ratio = self._resize_norm(img)
        x = paddle.to_tensor(padded[np.newaxis, :])
        vr = paddle.to_tensor(np.array([valid_ratio], dtype="float32"))

        with paddle.no_grad():
            feat = self.model.backbone(x)
            # SAR needs valid_ratio (real content width / padded width) via
            # img_metas=[labels_placeholder, valid_ratio] -- without it the
            # attention decoder doesn't know where the real plate ends and
            # keeps attending into the zero-padded region, hallucinating
            # extra characters past the true text (this is what made the
            # first, unfixed attempt at this produce garbage).
            sar_out = self.model.head.sar_head(feat, targets=[[""], vr])

        text, confidence = self.sar_post(sar_out)[0]
        return text, float(confidence)

    def recognize_double_line(self, top_row, bottom_row):
        """Mirrors PlateOCR.recognize_double_line's split-then-concatenate
        shape, but reads each row with SAR instead of CTC -- CTC's
        double-line reads kept the same blank-collapse/confusion problems
        SAR fixed for single-line crops (e.g. "AP39BD0606" -> "AP39BI0606",
        a plain character misread, not a join-boundary artifact)."""
        top_text, top_confidence = self.recognize(top_row)
        bottom_text, bottom_confidence = self.recognize(bottom_row)
        if not top_text and not bottom_text:
            return "", 0.0
        confidences = [c for c in (top_confidence, bottom_confidence) if c > 0]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return top_text + bottom_text, confidence
