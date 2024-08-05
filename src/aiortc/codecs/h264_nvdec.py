import itertools
import logging
from typing import List, Union, Optional

import av
import cvcuda
import numpy as np
import nvcv
import pycuda.driver as cuda
import PyNvVideoCodec as nvvc
import torch

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE
from .base import Decoder

logger = logging.getLogger(__name__)


class H264NvDecDecoder(Decoder):
    def __init__(self) -> None:
        device_id = 0
        batch_size = 1

        cuda_device = cuda.Device(device_id)
        self.cuda_ctx = cuda_device.retain_primary_context()
        self.cuda_ctx.push()

        cvcuda_stream = cvcuda.Stream()
        torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)

        self.cvcuda_stream = cvcuda_stream
        self.torch_stream = torch_stream
        self.codec = VideoBatchDecoder(
            batch_size, device_id, self.cuda_ctx, cvcuda_stream
        )

    def decode(self, encoded_frame: JitterFrame):
        try:
            packet = av.Packet(encoded_frame.data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE

            packet_data = nvvc.PacketData()
            packet_data.bsl = packet.buffer_size
            packet_data.bsl_data = packet.buffer_ptr
            packet_data.pts = packet.pts

            with self.cvcuda_stream, torch.cuda.stream(self.torch_stream):
                batch = self.codec(packet_data)

            return [batch.data] if batch else []
        except av.AVError as e:
            logger.warning(
                "H264NvDecDecoder() failed to decode, skipping package: " + str(e)
            )
            return []

    def stop(self):
        self.cuda_ctx.pop()


pixel_format_to_cvcuda_code = {
    nvvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,
    nvvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,
}

# Decoder impl from https://github.com/NVIDIA-AI-IOT/deepstream_libraries/blob/main/common/nvcodec_utils.py


class Batch:
    """
    This object helps us keep track of the data associated with a batch used
    throughout the deep learning pipelines of CVCUDA.
    In addition of tracking the data tensors associated with the batch, it
    allows tracking the index of the batch and any filename information one
    wants to attach (i.e. which files did the data come from).
    """

    def __init__(
        self,
        batch_idx: int,
        data: Union[cvcuda.Tensor, np.ndarray, torch.Tensor],
        fileinfo: Optional[Union[str, List[str]]] = "",
    ):
        """
        Initializes a new instance of the `Batch` class.
        :param batch_idx: A zero based int specifying the index of this batch.
        :param data: The data associated with this batch. Either a torch/CVCUDA tensor or a numpy array.
        :param fileinfo: Either a string or list or strings specifying any filename information of this batch.
        """
        self.batch_idx = batch_idx
        self.data = data
        self.fileinfo = fileinfo


class VideoBatchDecoder:
    def __init__(
        self,
        batch_size,
        device_id,
        cuda_ctx,
        cuda_stream,
    ):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream
        self.total_decoded = 0
        self.batch_idx = 0
        self.decoder = None
        self.cvcuda_RGBtensor_batch = None
        self.fps = 30
        self.logger.info("Using PyNvVideoCodec decoder version: %s" % nvvc.__version__)

    def __call__(self, packet: nvvc.PacketData):
        # Check if we need to allocate the decoder for its first use.
        if self.decoder is None:
            self.decoder = nvVideoDecoder(
                self.device_id, self.cuda_ctx, self.cuda_stream
            )

        # Get the NHWC YUV tensor from the decoder
        cvcuda_YUVtensor = self.decoder.get_next_frames(packet)

        # Check if we are done decoding
        if cvcuda_YUVtensor is None:
            return None

        # Check the code for the color conversion based in the pixel format
        cvcuda_code = pixel_format_to_cvcuda_code.get(self.decoder.pixelFormat)
        if cvcuda_code is None:
            raise ValueError(f"Unsupported pixel format: {self.decoder.pixelFormat}")

        # Check layout to make sure it is what we expected
        if cvcuda_YUVtensor.layout != "NHWC":
            raise ValueError("Unexpected tensor layout, NHWC expected.")

        # this may be different than batch size since last frames may not be a multiple of batch size
        actual_batch_size = cvcuda_YUVtensor.shape[0]

        # Create a CVCUDA tensor for color conversion YUV->RGB
        # Allocate only for the first time or for the last batch.
        if not self.cvcuda_RGBtensor_batch or actual_batch_size != self.batch_size:
            self.cvcuda_RGBtensor_batch = cvcuda.Tensor(
                (actual_batch_size, self.decoder.h, self.decoder.w, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

        # Convert from YUV to RGB. Conversion code is based on the pixel format.
        cvcuda.cvtcolor_into(self.cvcuda_RGBtensor_batch, cvcuda_YUVtensor, cvcuda_code)

        self.total_decoded += actual_batch_size

        # Create a batch instance and set its properties.
        batch = Batch(
            batch_idx=self.batch_idx,
            data=self.cvcuda_RGBtensor_batch,
        )
        self.batch_idx += 1

        return batch

    def start(self):
        pass

    def join(self):
        pass


class nvVideoDecoder:
    def __init__(self, device_id, cuda_ctx, stream):
        """
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        """
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.stream = stream
        self.nvDec = nvvc.CreateDecoder(
            gpuid=0,
            cudacontext=self.cuda_ctx.handle,
            cudastream=self.stream.handle,
            enableasyncallocations=False,
        )

        self.w, self.h = 512, 512
        self.pixelFormat = self.nvDec.GetPixelFormat()
        # In case sample aspect ratio isn't 1:1 we will re-scale the decoded
        # frame to maintain uniform 1:1 ratio across the pipeline.
        sar = 8.0 / 9.0
        self.fixed_h = self.h
        self.fixed_w = int(self.w * sar)

    # frame iterator
    def generate_decoded_frames(self, packet: nvvc.PacketData):
        for decodedFrame in self.nvDec.Decode(packet):
            nvcvTensor = nvcv.as_tensor(
                nvcv.as_image(decodedFrame.nvcv_image(), nvcv.Format.U8)
            )
            if nvcvTensor.layout == "NCHW":
                # This will re-format the NCHW tensor to a NHWC tensor which will create
                # a copy in the CUDA device decoded frame will go out of scope and the
                # backing memory will be available by the decoder.
                yield cvcuda.reformat(nvcvTensor, "NHWC")
            else:
                raise ValueError("Unexpected tensor layout, NCHW expected.")

    def get_next_frames(self, packet: nvvc.PacketData):
        decoded_frames = list(itertools.islice(self.generate_decoded_frames(packet), 1))
        if len(decoded_frames) == 0:
            return None
        elif len(decoded_frames) == 1:  # this case we dont need stack the tensor
            return decoded_frames[0]
        else:
            # convert from list of tensors to a single tensor (NHWC)
            tensorNHWC = cvcuda.stack(decoded_frames)
            return tensorNHWC
