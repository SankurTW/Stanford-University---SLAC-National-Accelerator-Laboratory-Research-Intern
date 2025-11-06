import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import struct
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationConfig:
    def __init__(
        self,
        weight_bits: int = 8,
        activation_bits: int = 8,
        use_symmetric: bool = True,
        per_channel: bool = True
    ):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.use_symmetric = use_symmetric
        self.per_channel = per_channel

class QuantizationAwareTraining:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.weight_scales = {}
        self.activation_scales = {}
   
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        num_bits: int,
        symmetric: bool = True
    ) -> Tuple[torch.Tensor, float]:
        if symmetric:
            max_val = torch.max(torch.abs(tensor))
            scale = (2 ** (num_bits - 1) - 1) / (max_val + 1e-8)
        else:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            scale = (2 ** num_bits - 1) / (max_val - min_val + 1e-8)
       
        quantized = torch.round(tensor * scale)
        quantized = torch.clamp(quantized, -2**(num_bits-1), 2**(num_bits-1)-1)
       
        return quantized, scale.item()
   
    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        return quantized / scale
   
    def quantize_model(self, model: nn.Module) -> Dict:
        quantized_params = {
            'layers': {},
            'config': {
                'weight_bits': self.config.weight_bits,
                'activation_bits': self.config.activation_bits
            }
        }
       
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_q, weight_scale = self.quantize_tensor(
                    module.weight.data,
                    self.config.weight_bits
                )
               
                bias_q, bias_scale = None, None
                if module.bias is not None:
                    bias_q, bias_scale = self.quantize_tensor(
                        module.bias.data,
                        self.config.weight_bits
                    )
               
                quantized_params['layers'][name] = {
                    'type': type(module).__name__,
                    'weight': weight_q.cpu().numpy(),
                    'weight_scale': weight_scale,
                    'bias': bias_q.cpu().numpy() if bias_q is not None else None,
                    'bias_scale': bias_scale,
                    'shape': list(module.weight.shape)
                }
               
                self.weight_scales[name] = weight_scale
       
        return quantized_params

class HLSCodeGenerator:
    def __init__(self, model_name: str = "ParticleNet"):
        self.model_name = model_name
        self.layer_code = []
   
    def generate_conv2d_hls(
        self,
        layer_name: str,
        weight: np.ndarray,
        bias: Optional[np.ndarray],
        stride: int,
        padding: int
    ) -> str:
        out_ch, in_ch, kh, kw = weight.shape
       
        code = f"""
// {layer_name}: Conv2D Layer
void {layer_name}(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream,
    const weight_t weights[{out_ch}][{in_ch}][{kh}][{kw}],
    const bias_t bias[{out_ch}]
) {{
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete
   
    data_t input_buffer[{in_ch}][MAX_HEIGHT][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
   
    data_t output_buffer[{out_ch}][MAX_HEIGHT][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=1
   
    LOAD_INPUT: for(int c = 0; c < {in_ch}; c++) {{
        for(int h = 0; h < MAX_HEIGHT; h++) {{
            for(int w = 0; w < MAX_WIDTH; w++) {{
                #pragma HLS PIPELINE II=1
                input_buffer[c][h][w] = in_stream.read();
            }}
        }}
    }}
   
    CONV_OUT_CH: for(int oc = 0; oc < {out_ch}; oc++) {{
        CONV_H: for(int h = 0; h < MAX_HEIGHT; h+={stride}) {{
            CONV_W: for(int w = 0; w < MAX_WIDTH; w+={stride}) {{
                #pragma HLS PIPELINE II=1
               
                acc_t accumulator = bias[oc];
               
                CONV_IN_CH: for(int ic = 0; ic < {in_ch}; ic++) {{
                    CONV_KH: for(int kh = 0; kh < {kh}; kh++) {{
                        CONV_KW: for(int kw = 0; kw < {kw}; kw++) {{
                            int h_idx = h + kh - {padding};
                            int w_idx = w + kw - {padding};
                           
                            if(h_idx >= 0 && h_idx < MAX_HEIGHT &&
                               w_idx >= 0 && w_idx < MAX_WIDTH) {{
                                accumulator += input_buffer[ic][h_idx][w_idx] *
                                              weights[oc][ic][kh][kw];
                            }}
                        }}
                    }}
                }}
               
                output_buffer[oc][h/{stride}][w/{stride}] = (accumulator > 0) ? accumulator : 0;
            }}
        }}
    }}
   
    STREAM_OUT: for(int c = 0; c < {out_ch}; c++) {{
        for(int h = 0; h < MAX_HEIGHT/{stride}; h++) {{
            for(int w = 0; w < MAX_WIDTH/{stride}; w++) {{
                #pragma HLS PIPELINE II=1
                out_stream.write(output_buffer[c][h][w]);
            }}
        }}
    }}
}}
"""
        return code
   
    def generate_linear_hls(
        self,
        layer_name: str,
        weight: np.ndarray,
        bias: Optional[np.ndarray]
    ) -> str:
        out_features, in_features = weight.shape
       
        code = f"""
// {layer_name}: Fully Connected Layer
void {layer_name}(
    const data_t input[{in_features}],
    data_t output[{out_features}],
    const weight_t weights[{out_features}][{in_features}],
    const bias_t bias[{out_features}]
) {{
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete
   
    FC_OUT: for(int o = 0; o < {out_features}; o++) {{
        #pragma HLS PIPELINE II=1
       
        acc_t accumulator = bias[o];
       
        FC_IN: for(int i = 0; i < {in_features}; i++) {{
            accumulator += input[i] * weights[o][i];
        }}
       
        output[o] = (accumulator > 0) ? accumulator : 0;
    }}
}}
"""
        return code
   
    def generate_complete_hls(
        self,
        quantized_params: Dict,
        input_shape: Tuple[int, ...]
    ) -> str:
        header = f"""
#ifndef {self.model_name.upper()}_H
#define {self.model_name.upper()}_H
#include "hls_stream.h"
#include "ap_fixed.h"
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<16, 6> weight_t;
typedef ap_fixed<32, 16> acc_t;
typedef ap_fixed<16, 6> bias_t;
#define MAX_HEIGHT {input_shape[2]}
#define MAX_WIDTH {input_shape[3]}
#define INPUT_CHANNELS {input_shape[1]}
#define NUM_CLASSES {quantized_params['config'].get('num_classes', 5)}
void {self.model_name}_inference(
    hls::stream<data_t> &input_stream,
    data_t output[NUM_CLASSES]
);
#endif
"""
       
        implementation = f"""
#include "{self.model_name}.h"
"""
       
        for layer_name, layer_params in quantized_params['layers'].items():
            if layer_params['type'] == 'Conv2d':
                self.layer_code.append(
                    self.generate_conv2d_hls(
                        layer_name.replace('.', '_'),
                        layer_params['weight'],
                        layer_params.get('bias'),
                        stride=1,
                        padding=1
                    )
                )
            elif layer_params['type'] == 'Linear':
                self.layer_code.append(
                    self.generate_linear_hls(
                        layer_name.replace('.', '_'),
                        layer_params['weight'],
                        layer_params.get('bias')
                    )
                )
       
        implementation += '\n'.join(self.layer_code)
       
        top_function = f"""
void {self.model_name}_inference(
    hls::stream<data_t> &input_stream,
    data_t output[NUM_CLASSES]
) {{
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE m_axi port=output depth=NUM_CLASSES
    #pragma HLS INTERFACE s_axilite port=return
   
    #pragma HLS DATAFLOW
   
    hls::stream<data_t> layer1_out;
    hls::stream<data_t> layer2_out;
    hls::stream<data_t> layer3_out;
}}
"""
        implementation += top_function
       
        return header, implementation

class FPGADeploymentPipeline:
    def __init__(
        self,
        target_platform: str = "xilinx_u250",
        target_latency_us: float = 1.0
    ):
        self.target_platform = target_platform
        self.target_latency = target_latency_us
        self.qat = None
   
    def prepare_model(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        num_bits: int = 8
    ) -> nn.Module:
        logger.info("Preparing model for FPGA deployment...")
       
        config = QuantizationConfig(
            weight_bits=num_bits,
            activation_bits=num_bits
        )
        self.qat = QuantizationAwareTraining(config)
       
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= 10:
                    break
                _ = model(data)
       
        logger.info("Model preparation complete")
        return model
   
    def export_for_fpga(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: Tuple[int, ...]
    ):
        logger.info("Exporting model for FPGA...")
       
        quantized_params = self.qat.quantize_model(model)
       
        codegen = HLSCodeGenerator(model_name="ParticleNet")
        header, implementation = codegen.generate_complete_hls(
            quantized_params,
            input_shape
        )
       
        import os
        os.makedirs(output_dir, exist_ok=True)
       
        with open(f"{output_dir}/ParticleNet.h", 'w') as f:
            f.write(header)
       
        with open(f"{output_dir}/ParticleNet.cpp", 'w') as f:
            f.write(implementation)
       
        np.savez(
            f"{output_dir}/quantized_weights.npz",
            ** 랜 {k: v['weight'] for k, v in quantized_params['layers'].items()}
        )
       
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump({
                'platform': self.target_platform,
                'target_latency_us': self.target_latency,
                'quantization': quantized_params['config']
            }, f, indent=2)
       
        logger.info(f"FPGA files exported to {output_dir}")
       
        self._generate_vitis_script(output_dir)
   
    def _generate_vitis_script(self, output_dir: str):
        script = """#!/bin/bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-tensorflow
vai_c_tensorflow \\
    --frozen_pb quantized_model.pb \\
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/U50/arch.json \\
    --output_dir compiled_model \\
    --net_name particle_net \\
    --options "{'mode':'normal'}"
echo "FPGA compilation complete!"
echo "Deploy compiled_model/particle_net.xmodel to FPGA"
"""
       
        with open(f"{output_dir}/compile_for_fpga.sh", 'w') as f:
            f.write(script)
       
        import os
        os.chmod(f"{output_dir}/compile_for_fpga.sh", 0o755)

class FPGAInferenceEngine:
    def __init__(self, quantized_params: Dict):
        self.params = quantized_params
        self.layers = {}
        self._build_engine()
   
    def _build_engine(self):
        for name, layer_params in self.params['layers'].items():
            self.layers[name] = {
                'weight': torch.tensor(layer_params['weight'], dtype=torch.float32),
                'bias': torch.tensor(layer_params['bias'], dtype=torch.float32)
                        if layer_params['bias'] is not None else None,
                'scale': layer_params['weight_scale']
            }
   
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        x = torch.tensor(input_data, dtype=torch.float32)
       
        for name, layer in self.layers.items():
            weight = layer['weight'] / layer['scale']
            pass
       
        return x.numpy()

def profile_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100
) -> Dict[str, float]:
    import time
   
    model.eval()
    dummy_input = torch.randn(*input_shape)
   
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
   
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
   
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
   
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
   
    avg_latency_ms = (end - start) / num_iterations * 1000
    estimated_fpga_latency_us = avg_latency_ms * 20
   
    return {
        'cpu_latency_ms': avg_latency_ms,
        'estimated_fpga_latency_us': estimated_fpga_latency_us,
        'throughput_hz': 1000 / avg_latency_ms
    }

if __name__ == "__main__":
    logger.info("FPGA Acceleration Pipeline for SLAC Research")
   
    from cnn_detector import FPGAOptimizedCNN
   
    model = FPGAOptimizedCNN(num_input_channels=1, num_classes=5)
   
    latency_stats = profile_latency(model, (1, 1, 128, 128))
    logger.info(f"Latency Profile: {latency_stats}")
   
    pipeline = FPGADeploymentPipeline(
        target_platform="xilinx_u250",
        target_latency_us=1.0
    )
   
    logger.info("FPGA pipeline ready for model deployment")
    logger.info("Use pipeline.export_for_fpga() to generate HLS code")