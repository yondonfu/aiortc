import logging
import math
import os
from fractions import Fraction
from struct import pack
from typing import Iterator, List, Optional, Tuple

import av
import cvcuda
import numpy as np
import nvcv
import PyNvVideoCodec as nvvc
import torch
from av.frame import Frame
from av.packet import Packet

from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Encoder

DEFAULT_BITRATE = int(os.getenv("NVENC_DEFAULT_BITRATE", 1000000))  # default 1 mbps
MIN_BITRATE = int(os.getenv("NVENC_MIN_BITRATE", 500000))  # default 500 kbps
MAX_BITRATE = int(os.getenv("NVENC_MAX_BITRATE", 1000000))  # default 1 mpbs
FRAMERATE = int(os.getenv("NVENC_FRAMERATE", 30))
TUNING_INFO = os.getenv("NVENC_TUNING_INFO", "high_quality")
PRESET = os.getenv("NVENC_PRESET", "P4")

PACKET_MAX = 1300

NAL_TYPE_FU_A = 28
NAL_TYPE_STAP_A = 24

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2
LENGTH_FIELD_SIZE = 2
STAP_A_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE


class H264NvEncEncoder(Encoder):
    def __init__(self) -> None:
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None
        self.codec = None
        self.codec_buffering = False
        self.__target_bitrate = DEFAULT_BITRATE
        self.pts_time = 0
        self.delta_t = 1

    @staticmethod
    def _packetize_fu_a(data: bytes) -> List[bytes]:
        available_size = PACKET_MAX - FU_A_HEADER_SIZE
        payload_size = len(data) - NAL_HEADER_SIZE
        num_packets = math.ceil(payload_size / available_size)
        num_larger_packets = payload_size % num_packets
        package_size = payload_size // num_packets

        f_nri = data[0] & (0x80 | 0x60)  # fni of original header
        nal = data[0] & 0x1F

        fu_indicator = f_nri | NAL_TYPE_FU_A

        fu_header_end = bytes([fu_indicator, nal | 0x40])
        fu_header_middle = bytes([fu_indicator, nal])
        fu_header_start = bytes([fu_indicator, nal | 0x80])
        fu_header = fu_header_start

        packages = []
        offset = NAL_HEADER_SIZE
        while offset < len(data):
            if num_larger_packets > 0:
                num_larger_packets -= 1
                payload = data[offset : offset + package_size + 1]
                offset += package_size + 1
            else:
                payload = data[offset : offset + package_size]
                offset += package_size

            if offset == len(data):
                fu_header = fu_header_end

            packages.append(fu_header + payload)

            fu_header = fu_header_middle
        assert offset == len(data), "incorrect fragment data"

        return packages

    @staticmethod
    def _packetize_stap_a(
        data: bytes, packages_iterator: Iterator[bytes]
    ) -> Tuple[bytes, bytes]:
        counter = 0
        available_size = PACKET_MAX - STAP_A_HEADER_SIZE

        stap_header = NAL_TYPE_STAP_A | (data[0] & 0xE0)

        payload = bytes()
        try:
            nalu = data  # with header
            while len(nalu) <= available_size and counter < 9:
                stap_header |= nalu[0] & 0x80

                nri = nalu[0] & 0x60
                if stap_header & 0x60 < nri:
                    stap_header = stap_header & 0x9F | nri

                available_size -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(packages_iterator)

            if counter == 0:
                nalu = next(packages_iterator)
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        else:
            return bytes([stap_header]) + payload, nalu

    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        # Translated from: https://github.com/aizvorski/h264bitstream/blob/master/h264_nal.c#L134
        i = 0
        while True:
            # Find the start of the NAL unit.
            #
            # NAL Units start with the 3-byte start code 0x000001 or
            # the 4-byte start code 0x00000001.
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                return

            # Jump past the start code
            i += 3
            nal_start = i

            # Find the end of the NAL unit (end of buffer OR next start code)
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                yield buf[nal_start : len(buf)]
                return
            elif buf[i - 1] == 0:
                # 4-byte start code case, jump back one byte
                yield buf[nal_start : i - 1]
            else:
                yield buf[nal_start:i]

    @classmethod
    def _packetize(cls, packages: Iterator[bytes]) -> List[bytes]:
        packetized_packages = []

        packages_iterator = iter(packages)
        package = next(packages_iterator, None)
        while package is not None:
            if len(package) > PACKET_MAX:
                packetized_packages.extend(cls._packetize_fu_a(package))
                package = next(packages_iterator, None)
            else:
                packetized, package = cls._packetize_stap_a(package, packages_iterator)
                packetized_packages.append(packetized)

        return packetized_packages

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        # if self.codec and (
        #     frame.width != self.codec.width
        #     or frame.height != self.codec.height
        #     # we only adjust bitrate if it changes by over 10%
        #     or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
        #     > 0.1
        # ):
        #     self.buffer_data = b""
        #     self.buffer_pts = None
        #     self.codec = None

        # if force_keyframe:
        #     # force a complete image
        #     frame.pict_type = av.video.frame.PictureType.I
        # else:
        #     # reset the picture type, otherwise no B-frames are produced
        #     frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            self.codec = VideoBatchEncoder(self.target_bitrate)

        data_to_send = b""
        for package in self.codec(frame):
            package_bytes = bytes(package)
            if self.codec_buffering:
                # delay sending to ensure we accumulate all packages
                # for a given PTS
                if package.pts == self.buffer_pts:
                    self.buffer_data += package_bytes
                else:
                    data_to_send += self.buffer_data
                    self.buffer_data = package_bytes
                    self.buffer_pts = package.pts
            else:
                data_to_send += package_bytes

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        assert isinstance(frame, torch.Tensor)
        packages = self._encode_frame(frame, force_keyframe)
        timestamp = convert_timebase(self.pts_time, 1 / Fraction(30), VIDEO_TIME_BASE)

        self.pts_time += self.delta_t

        return self._packetize(packages), timestamp

    def pack(self, packet: Packet) -> Tuple[List[bytes], int]:
        assert isinstance(packet, av.Packet)
        packages = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    @property
    def target_bitrate(self) -> int:
        """
        Target bitrate in bits per second.
        """
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))
        self.__target_bitrate = bitrate

    def stop(self):
        self.codec.join()


# Encoder impl from https://github.com/NVIDIA-AI-IOT/deepstream_libraries/blob/main/common/nvcodec_utils.py


class VideoBatchEncoder:
    def __init__(self, bitrate):
        self.logger = logging.getLogger(__name__)
        self.cuda_stream = cvcuda.Stream().current
        self.encoder = None
        self.cvcuda_HWCtensor_batch = None
        self.cvcuda_YUVtensor_batch = None
        self.input_layout = "NCHW"
        self.gpu_input = True
        self.bitrate = bitrate

        self.logger.info("Using PyNvVideoCodec encoder version: %s" % nvvc.__version__)

    def __call__(self, batch):
        assert isinstance(batch.data, torch.Tensor)

        # Check if we need to allocate the encoder for its first use.
        if self.encoder is None:
            self.encoder = nvVideoEncoder(
                batch.data.shape[3],
                batch.data.shape[2],
                self.cuda_stream,
                "NV12",
                self.bitrate,
            )

        # Create 2 CVCUDA tensors: reformat NCHW->NHWC and color conversion RGB->YUV
        current_batch_size = batch.data.shape[0]
        height, width = batch.data.shape[2], batch.data.shape[3]

        # Allocate only for the first time or for the last batch.
        if (
            not self.cvcuda_HWCtensor_batch
            or current_batch_size != self.cvcuda_HWCtensor_batch.shape[0]
        ):
            self.cvcuda_HWCtensor_batch = cvcuda.Tensor(
                (current_batch_size, height, width, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )
            self.cvcuda_YUVtensor_batch = cvcuda.Tensor(
                (current_batch_size, (height // 2) * 3, width, 1),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

        # Convert RGB to NV12, in batch, before sending it over to pyVideoCodec.
        # Convert to CVCUDA tensor
        cvcuda_tensor = cvcuda.as_tensor(batch.data, nvcv.TensorLayout.NCHW)

        # Reformat NCHW to NHWC
        cvcuda.reformat_into(self.cvcuda_HWCtensor_batch, cvcuda_tensor)

        # Color convert from RGB to YUV_NV12
        cvcuda.cvtcolor_into(
            self.cvcuda_YUVtensor_batch,
            self.cvcuda_HWCtensor_batch,
            cvcuda.ColorConversion.RGB2YUV_NV12,
        )

        # Convert back to torch tensor we are NV12
        tensor = torch.as_tensor(self.cvcuda_YUVtensor_batch.cuda(), device="cuda")

        # Encode frames from the batch one by one using pyVideoCodec.
        for img_idx in range(tensor.shape[0]):
            img = tensor[img_idx]
            # WAR sync the cvcuda active stream since encoder is on default
            cvcuda.Stream.current.sync()  # WAR
            yield self.encoder.encode_from_tensor(img)

    def start(self):
        pass

    def join(self):
        self.encoder.flush()


class nvVideoEncoder:
    def __init__(self, width, height, cuda_stream, format, bitrate):
        """
        Create instance of HW-accelerated video encoder.
        :param width: encoded frame width.
        :param height: encoded frame height.
        :param format: The format of the encoded video file.
                (e.g. "NV12", "YUV444" see NvPyVideoEncoder docs for more info)
        """
        self.logger = logging.getLogger(__name__)
        self.cuda_stream = cuda_stream

        self.pts_time = 0
        self.delta_t = 1
        self.encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)

        aligned_value = 0
        if width % 16 != 0:
            aligned_value = 16 - (width % 16)
        aligned_width = width + aligned_value
        width = aligned_width

        config = {
            "preset": PRESET,
            "codec": "h264",
            "cudastream": cuda_stream.handle,
            "fps": FRAMERATE,
            "tuning_info": TUNING_INFO,
            "bitrate": bitrate,
            "maxbitrate": bitrate,
        }
        self.logger.info(
            f"creating NVENC width={width} height={height} format={format}"
        )
        self.logger.info(f"**kwargs={config}")

        self.nvEnc = nvvc.CreateEncoder(width, height, format, False, **config)

    def width(self):
        """
        Gets the actual video frame width from the encoder.
        """
        return self.nvEnc.Width()

    def height(self):
        """
        Gets the actual video frame height from the encoder.
        """
        return self.nvEnc.Height()

    def encode_from_tensor(self, tensor):
        # Encode the frame takes tensor as input
        self.encoded_frame = self.nvEnc.Encode(tensor)

        encoded_bytes = bytearray(self.encoded_frame)
        packet = av.packet.Packet(encoded_bytes)
        packet.pts = self.pts_time
        packet.dts = self.pts_time
        packet.time_base = 1 / Fraction(30)

        self.pts_time += self.delta_t

        return packet

    def flush(self):
        self.nvEnc.EndEncode()
