"""Unit tests for the AMD AMF hardware-encoding paths.

Mirrors tests/test_nvenc.py — exercises the config matrix, the encoder-
argument builder, and the availability probe's graceful-failure behaviour.
None of these need a real ffmpeg or an AMD GPU present.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    ENCODER_KEY_MAP, ENCODER_OPTIONS, VIDEO_FORMATS, VIDEO_QUALITY_PRESETS,
    amf_available, build_video_encoder_args,
)


def _val(args, flag):
    """Return the token following `flag` in an ffmpeg arg list, or None."""
    for i, tok in enumerate(args[:-1]):
        if tok == flag:
            return args[i + 1]
    return None


class ConfigConsistencyTest(unittest.TestCase):
    def test_every_amf_format_has_preset_quality_values(self):
        for name, fmt in VIDEO_FORMATS.items():
            key = fmt.get("amf_key")
            if key is None:
                continue
            for preset_name, preset in VIDEO_QUALITY_PRESETS.items():
                self.assertIn(
                    key, preset,
                    f"{preset_name} missing AMF quality value {key} "
                    f"needed by format {name}",
                )

    def test_every_preset_has_amf_quality(self):
        for preset_name, preset in VIDEO_QUALITY_PRESETS.items():
            self.assertIn("amf_quality", preset, preset_name)
            self.assertRegex(preset["amf_quality"],
                             r"^(speed|balanced|quality)$")

    def test_only_vp9_lacks_an_amf_codec(self):
        for name, fmt in VIDEO_FORMATS.items():
            if fmt["codec_key"] == "vp9":
                self.assertIsNone(fmt["amf_codec"], name)
            else:
                self.assertTrue(fmt["amf_codec"].endswith("_amf"), name)

    def test_encoder_key_map_includes_amf(self):
        self.assertIn("AMD GPU (AMF)", ENCODER_OPTIONS)
        self.assertEqual(ENCODER_KEY_MAP["AMD GPU (AMF)"], "amf")
        # AMF must be the 4th option so _apply_amf_availability's index [3]
        # lines up with the UI.
        self.assertEqual(ENCODER_OPTIONS[3], "AMD GPU (AMF)")

    def test_nvenc_and_amf_presets_align_in_quality_ladder(self):
        # Whatever Cove calls the slowest preset should pick the highest-fidelity
        # AMF quality mode; the fastest preset should pick "speed".
        ordered = list(VIDEO_QUALITY_PRESETS.values())
        for p in ordered:
            self.assertIn(p["amf_quality"],
                          ("speed", "balanced", "quality"))
        self.assertEqual(ordered[0]["amf_quality"], "speed")
        self.assertEqual(ordered[-1]["amf_quality"], "quality")


class EncoderArgsAmfTest(unittest.TestCase):
    def test_amf_quality_mode_uses_cqp_not_cq_or_crf(self):
        args = build_video_encoder_args(
            encoder="hevc_amf", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=None, crf=28, speed_preset="medium",
            nvenc_preset="p6", amf_quality="balanced",
        )
        self.assertEqual(_val(args, "-c:v"), "hevc_amf")
        self.assertEqual(_val(args, "-quality"), "balanced")
        self.assertEqual(_val(args, "-usage"), "transcoding")
        self.assertEqual(_val(args, "-rc"), "cqp")
        self.assertEqual(_val(args, "-qp"), "28")
        self.assertNotIn("-crf", args)
        self.assertNotIn("-cq", args)
        # AMF must never take the log-file two-pass path.
        self.assertNotIn("-pass", args)

    def test_amf_bitrate_mode_uses_vbr_with_capped_maxrate(self):
        args = build_video_encoder_args(
            encoder="h264_amf", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=2000, crf=None, speed_preset="medium",
            nvenc_preset="p5", amf_quality="speed",
        )
        self.assertEqual(_val(args, "-b:v"), "2000k")
        self.assertEqual(_val(args, "-maxrate"), "2800k")   # 2000 * 1.4
        self.assertEqual(_val(args, "-bufsize"), "4000k")   # 2000 * 2
        self.assertEqual(_val(args, "-rc"), "vbr")
        # AMF has no equivalent of NVENC's -multipass fullres.
        self.assertNotIn("-multipass", args)
        self.assertNotIn("-pass", args)

    def test_amf_applies_scale_filter(self):
        args = build_video_encoder_args(
            encoder="hevc_amf", vf="scale=1280:1280", use_two_pass=False,
            pass_num=None, video_kbps=None, crf=25, speed_preset="slow",
            nvenc_preset="p7", amf_quality="quality",
        )
        self.assertEqual(_val(args, "-vf"), "scale=1280:1280")
        self.assertEqual(_val(args, "-quality"), "quality")

    def test_amf_quality_dial_default_is_balanced(self):
        # Cover the default-arg path: compress_video passes amf_quality
        # explicitly, but callers without it must still get a sane encoder.
        args = build_video_encoder_args(
            encoder="h264_amf", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=None, crf=26, speed_preset="medium",
            nvenc_preset="p6",
        )
        self.assertEqual(_val(args, "-quality"), "balanced")


class AmfAvailabilityTest(unittest.TestCase):
    def test_missing_ffmpeg_reports_unavailable(self):
        saved_bin = compressor.FFMPEG_BIN
        compressor.FFMPEG_BIN = "cove-nonexistent-ffmpeg-binary"
        compressor._amf_cache.clear()
        try:
            self.assertFalse(amf_available("hevc_amf"))
            # Second call is served from cache — still False, no exception.
            self.assertFalse(amf_available("hevc_amf"))
            self.assertIn("hevc_amf", compressor._amf_cache)
        finally:
            compressor.FFMPEG_BIN = saved_bin
            compressor._amf_cache.clear()


if __name__ == "__main__":
    unittest.main()