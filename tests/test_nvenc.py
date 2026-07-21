"""Unit tests for the NVENC hardware-encoding paths.

These exercise the config matrix, the encoder-argument builder, and the
availability probe's graceful-failure behaviour — none of which need a real
ffmpeg or an NVIDIA GPU present.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    ENCODER_KEY_MAP, ENCODER_OPTIONS, VIDEO_FORMATS, VIDEO_QUALITY_PRESETS,
    build_video_encoder_args, nvenc_available,
)


def _val(args, flag):
    """Return the token following `flag` in an ffmpeg arg list, or None."""
    for i, tok in enumerate(args[:-1]):
        if tok == flag:
            return args[i + 1]
    return None


class ConfigConsistencyTest(unittest.TestCase):
    def test_every_nvenc_format_has_preset_quality_values(self):
        for name, fmt in VIDEO_FORMATS.items():
            key = fmt.get("nvenc_key")
            if key is None:
                continue
            for preset_name, preset in VIDEO_QUALITY_PRESETS.items():
                self.assertIn(
                    key, preset,
                    f"{preset_name} missing NVENC quality value {key} "
                    f"needed by format {name}",
                )

    def test_every_preset_has_speed_and_nvenc_preset(self):
        for preset_name, preset in VIDEO_QUALITY_PRESETS.items():
            self.assertIn("speed", preset, preset_name)
            self.assertIn("nvenc_preset", preset, preset_name)
            self.assertRegex(preset["nvenc_preset"], r"^p[1-7]$")

    def test_only_vp9_lacks_an_nvenc_codec(self):
        for name, fmt in VIDEO_FORMATS.items():
            if fmt["codec_key"] == "vp9":
                self.assertIsNone(fmt["nvenc_codec"], name)
            else:
                self.assertTrue(fmt["nvenc_codec"].endswith("_nvenc"), name)

    def test_encoder_key_map_covers_all_options(self):
        self.assertEqual(set(ENCODER_KEY_MAP), set(ENCODER_OPTIONS))
        self.assertEqual(
            set(ENCODER_KEY_MAP.values()), {"auto", "cpu", "nvenc", "amf"}
        )


class EncoderArgsNvencTest(unittest.TestCase):
    def test_nvenc_quality_mode_uses_cq_not_crf(self):
        args = build_video_encoder_args(
            encoder="hevc_nvenc", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=None, crf=27, speed_preset="medium", nvenc_preset="p6",
        )
        self.assertEqual(_val(args, "-c:v"), "hevc_nvenc")
        self.assertEqual(_val(args, "-preset"), "p6")
        self.assertEqual(_val(args, "-tune"), "hq")
        self.assertEqual(_val(args, "-rc"), "vbr")
        self.assertEqual(_val(args, "-cq"), "27")
        self.assertEqual(_val(args, "-b:v"), "0")
        self.assertNotIn("-crf", args)

    def test_nvenc_bitrate_mode_uses_single_pass_multipass(self):
        args = build_video_encoder_args(
            encoder="h264_nvenc", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=2000, crf=None, speed_preset="medium", nvenc_preset="p5",
        )
        self.assertEqual(_val(args, "-b:v"), "2000k")
        self.assertEqual(_val(args, "-maxrate"), "2800k")   # 2000 * 1.4
        self.assertEqual(_val(args, "-bufsize"), "4000k")   # 2000 * 2
        self.assertEqual(_val(args, "-multipass"), "fullres")
        self.assertEqual(_val(args, "-rc"), "vbr")
        # NVENC must never take the log-file two-pass path.
        self.assertNotIn("-pass", args)

    def test_nvenc_applies_scale_filter(self):
        args = build_video_encoder_args(
            encoder="hevc_nvenc", vf="scale=1280:1280", use_two_pass=False,
            pass_num=None, video_kbps=None, crf=24, speed_preset="slow",
            nvenc_preset="p7",
        )
        self.assertEqual(_val(args, "-vf"), "scale=1280:1280")


class EncoderArgsCpuUnchangedTest(unittest.TestCase):
    """The software paths must be byte-for-byte what they were pre-NVENC."""

    def test_x265_quality(self):
        args = build_video_encoder_args(
            encoder="libx265", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=None, crf=25, speed_preset="medium", nvenc_preset="p6",
        )
        self.assertEqual(
            args,
            ["-c:v", "libx265", "-preset", "medium", "-crf", "25",
             "-x265-params", "log-level=error"],
        )

    def test_x264_two_pass_first_pass(self):
        args = build_video_encoder_args(
            encoder="libx264", vf=None, use_two_pass=True, pass_num=1,
            video_kbps=1500, crf=None, speed_preset="medium", nvenc_preset="p6",
        )
        self.assertEqual(
            args,
            ["-c:v", "libx264", "-preset", "medium", "-b:v", "1500k",
             "-pass", "1"],
        )

    def test_vp9_quality_keeps_constant_quality_zero_bitrate(self):
        args = build_video_encoder_args(
            encoder="libvpx-vp9", vf=None, use_two_pass=False, pass_num=None,
            video_kbps=None, crf=31, speed_preset="medium", nvenc_preset="p6",
        )
        self.assertEqual(
            args,
            ["-c:v", "libvpx-vp9", "-row-mt", "1", "-cpu-used", "4",
             "-crf", "31", "-b:v", "0"],
        )


class NvencAvailabilityTest(unittest.TestCase):
    def test_missing_ffmpeg_reports_unavailable(self):
        saved_bin = compressor.FFMPEG_BIN
        compressor.FFMPEG_BIN = "cove-nonexistent-ffmpeg-binary"
        compressor._nvenc_cache.clear()
        try:
            self.assertFalse(nvenc_available("hevc_nvenc"))
            # Second call is served from cache — still False, no exception.
            self.assertFalse(nvenc_available("hevc_nvenc"))
            self.assertIn("hevc_nvenc", compressor._nvenc_cache)
        finally:
            compressor.FFMPEG_BIN = saved_bin
            compressor._nvenc_cache.clear()


if __name__ == "__main__":
    unittest.main()
