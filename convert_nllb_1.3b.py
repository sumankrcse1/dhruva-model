#!/usr/bin/env python3
"""
Python script to convert and split Meta's NLLB-200 Distilled 1.3B model into the optimized
ONNX format required for Dhruva-Translate / RTranslator (avoiding weight duplication).

Prerequisites:
    pip install optimum[onnxruntime] onnx onnxruntime
"""

import os
import sys
import subprocess
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_base_onnx():
    print("=== Step 1: Exporting NLLB-200 1.3B to ONNX via Hugging Face Optimum ===")

    def apply_compatibility_patches():
        # Apply compatibility monkeypatch for Python 3.13+ / 3.14+
        try:
            import optimum.exporters.base
            def patched_init(self, config, task, int_dtype="int64", float_dtype="fp32"):
                self.task = task
                self._config = config
                self._normalized_config = self.__class__.NORMALIZED_CONFIG_CLASS(self._config)
                self.int_dtype = int_dtype
                self.float_dtype = float_dtype
            optimum.exporters.base.ExporterConfig.__init__ = patched_init
            print("Successfully applied Python 3.13+ compatibility patch to ExporterConfig.__init__")
        except Exception as e:
            print(f"Note: Could not apply Python 3.13+ compatibility patch: {e}")

        # Apply memory optimization monkeypatch to load model with low_cpu_mem_usage=True
        try:
            import optimum.exporters.tasks
            original_get_model = optimum.exporters.tasks.TasksManager.get_model_from_task
            @classmethod
            def patched_get_model(cls, task, model_name_or_path, **kwargs):
                kwargs["low_cpu_mem_usage"] = True
                kwargs["use_safetensors"] = True
                return original_get_model(task, model_name_or_path, **kwargs)
            optimum.exporters.tasks.TasksManager.get_model_from_task = patched_get_model
            print("Successfully applied memory optimization patch (low_cpu_mem_usage) and forced safetensors.")
        except Exception as e:
            print(f"Note: Could not apply memory optimization patch: {e}")

        # Apply memory optimization patch for ONNX runtime inference session to prevent OOM
        try:
            import gc
            import torch
            import optimum.exporters.onnx.base
            original_fix = optimum.exporters.onnx.base.OnnxConfig.fix_dynamic_axes
            def patched_fix(self, model_path, device="cpu", dtype=None, input_shapes=None):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return original_fix(self, model_path, device, dtype, input_shapes)
            optimum.exporters.onnx.base.OnnxConfig.fix_dynamic_axes = patched_fix
            print("Successfully applied dynamic axes garbage collection patch.")
        except Exception as e:
            print(f"Note: Could not patch fix_dynamic_axes: {e}")

    def run_main_export():
        try:
            from optimum.exporters.onnx import main_export
            print("Exporting model programmatically via optimum.exporters.onnx.main_export...")
            main_export(
                model_name_or_path="facebook/nllb-200-distilled-1.3B",
                output="nllb_base_onnx/",
                task="seq2seq-lm-with-past",
                do_validation=False,
                no_post_process=True
            )
        except Exception as e:
            print(f"Programmatic export failed: {e}")
            raise e

    if "--export-encoder" in sys.argv:
        print("Subprocess: Starting encoder export...")
        apply_compatibility_patches()
        try:
            import optimum.exporters.onnx.convert
            original_export_models = optimum.exporters.onnx.convert.export_models
            def patched_export_models(models_and_onnx_configs, output_dir, opset=None, output_names=None, **kwargs):
                new_configs = {k: v for k, v in models_and_onnx_configs.items() if k == "encoder_model"}
                if output_names is not None:
                    output_names = ["encoder_model.onnx"]
                print(f"Patched export_models (Encoder): exporting {list(new_configs.keys())}")
                return original_export_models(new_configs, output_dir, opset=opset, output_names=output_names, **kwargs)
            optimum.exporters.onnx.convert.export_models = patched_export_models
        except Exception as e:
            print(f"Error patching: {e}")
        run_main_export()
        sys.exit(0)

    elif "--export-decoder" in sys.argv:
        print("Subprocess: Starting decoder export...")
        apply_compatibility_patches()
        try:
            import optimum.exporters.onnx.convert
            original_export_models = optimum.exporters.onnx.convert.export_models
            def patched_export_models(models_and_onnx_configs, output_dir, opset=None, output_names=None, **kwargs):
                new_configs = {}
                for k, v in models_and_onnx_configs.items():
                    if k == "decoder_with_past_model":
                        new_configs["decoder_model"] = v
                if output_names is not None:
                    output_names = ["decoder_model.onnx"]
                print(f"Patched export_models (Decoder): exporting {list(new_configs.keys())}")
                return original_export_models(new_configs, output_dir, opset=opset, output_names=output_names, **kwargs)
            optimum.exporters.onnx.convert.export_models = patched_export_models
        except Exception as e:
            print(f"Error patching: {e}")
        run_main_export()
        sys.exit(0)

    else:
        print("Launching encoder export in separate process...")
        subprocess.run([sys.executable, sys.argv[0], "--export-encoder"], check=True)
        print("Encoder export complete. Launching decoder export in separate process...")
        subprocess.run([sys.executable, sys.argv[0], "--export-decoder"], check=True)
        print("Base ONNX models successfully exported to nllb_base_onnx/\n")

def find_embedding_weight(model):
    # Search for the word embedding matrix: vocab_size is 256206, hidden_size is 1024
    for init in model.graph.initializer:
        if init.dims == [256206, 1024] or init.dims == [1024, 256206]:
            return init
    return None

def find_attention_weight_name(initializers, layer_idx, proj_type):
    # proj_type is 'k_proj' or 'v_proj'
    weight_name = None
    bias_name = None
    for init in initializers:
        name = init.name
        # Matches layer index and proj_type (e.g., layers.i.encoder_attn.k_proj)
        if f'layers.{layer_idx}.' in name and f'encoder_attn.{proj_type}' in name:
            if name.endswith('.weight') or name.endswith('_weight'):
                weight_name = name
            elif name.endswith('.bias') or name.endswith('_bias'):
                bias_name = name
    return weight_name, bias_name

def build_embed_and_lm_head(embed_weight_initializer, output_path):
    print("Building NLLB_embed_and_lm_head.onnx...")
    # Inputs
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.INT64, ['batch', 'seq'])
    pre_logits = helper.make_tensor_value_info('pre_logits', TensorProto.FLOAT, ['batch', 'seq', 1024])
    use_lm_head = helper.make_tensor_value_info('use_lm_head', TensorProto.BOOL, [])
    
    # Outputs
    embed_matrix = helper.make_tensor_value_info('embed_matrix', TensorProto.FLOAT, ['batch', 'seq', 1024])
    logits = helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'seq', 256206])
    
    # Scale factor for NLLB/BART embedding: sqrt(hidden_size) = sqrt(1024) = 32.0
    scale_val = helper.make_tensor('scale_val', TensorProto.FLOAT, [1], [32.0])
    
    # Nodes
    # 1. Gather (lookup token embeddings)
    gather_node = helper.make_node(
        'Gather',
        inputs=[embed_weight_initializer.name, 'input_ids'],
        outputs=['embed_unscaled'],
        axis=0
    )
    # 2. Multiply by scaling factor
    mul_node = helper.make_node(
        'Mul',
        inputs=['embed_unscaled', 'scale_val'],
        outputs=['embed_matrix']
    )
    # 3. Transpose weights for output projection (weight tying)
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[embed_weight_initializer.name],
        outputs=['weight_transposed'],
        perm=[1, 0]
    )
    # 4. MatMul for LM projection head
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['pre_logits', 'weight_transposed'],
        outputs=['logits']
    )
    
    # Assemble Graph
    graph = helper.make_graph(
        nodes=[gather_node, mul_node, transpose_node, matmul_node],
        name='embed_and_lm_head',
        inputs=[input_ids, pre_logits, use_lm_head],
        outputs=[embed_matrix, logits],
        initializer=[embed_weight_initializer, scale_val]
    )
    
    model = helper.make_model(graph, producer_name='dhruva-converter', ir_version=10, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.save(model, output_path)
    print(f"Created model: {output_path}")

def prune_encoder_embedding(model_path, output_path, embed_weight_name):
    print(f"Pruning embedding weights from encoder: {model_path} -> {output_path}")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Add embed_matrix as input to the graph
    embed_matrix_value_info = helper.make_tensor_value_info(
        'embed_matrix', 
        TensorProto.FLOAT, 
        ['batch', 'seq', 1024]
    )
    graph.input.append(embed_matrix_value_info)
    
    # Locate Gather node that queries input_ids using the embedding weight name
    embed_node = None
    for node in graph.node:
        if node.op_type == 'Gather' and node.input[0] == embed_weight_name:
            embed_node = node
            break
            
    if embed_node is not None:
        # Find subsequent Mul scaling node
        mul_node = None
        for node in graph.node:
            if node.op_type == 'Mul' and embed_node.output[0] in node.input:
                mul_node = node
                break
                
        target_tensor_name = mul_node.output[0] if mul_node else embed_node.output[0]
        
        # Rewire consumers to use our new embed_matrix input instead
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == target_tensor_name:
                    node.input[i] = 'embed_matrix'
                    
        # Remove embedding nodes from the graph
        if mul_node:
            graph.node.remove(mul_node)
        graph.node.remove(embed_node)
        
        # Prune the original shared embedding weights initializer from graph
        for init in list(graph.initializer):
            if init.name == embed_weight_name:
                graph.initializer.remove(init)
                break
                
    onnx.save(model, output_path, save_as_external_data=True, all_tensors_to_one_file=True)
    print(f"Pruned encoder saved successfully.")

def prune_decoder_embedding_and_lm_head(model_path, output_path, embed_weight_name):
    print(f"Pruning embeddings and projection from decoder: {model_path} -> {output_path}")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Add embed_matrix as input to the graph
    embed_matrix_value_info = helper.make_tensor_value_info(
        'embed_matrix', 
        TensorProto.FLOAT, 
        ['batch', 'seq', 1024]
    )
    graph.input.append(embed_matrix_value_info)
    
    # 1. Prune decoder embedding nodes
    embed_node = None
    for node in graph.node:
        if node.op_type == 'Gather' and node.input[0] == embed_weight_name:
            embed_node = node
            break
            
    if embed_node is not None:
        mul_node = None
        for node in graph.node:
            if node.op_type == 'Mul' and embed_node.output[0] in node.input:
                mul_node = node
                break
                
        target_tensor_name = mul_node.output[0] if mul_node else embed_node.output[0]
        
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == target_tensor_name:
                    node.input[i] = 'embed_matrix'
                    
        if mul_node:
            graph.node.remove(mul_node)
        graph.node.remove(embed_node)
        
        for init in list(graph.initializer):
            if init.name == embed_weight_name:
                graph.initializer.remove(init)
                break
                
    # 2. Prune decoder LM Head
    lm_head_node = None
    for node in graph.node:
        if 'logits' in node.output:
            lm_head_node = node
            break
            
    if lm_head_node is not None:
        pre_logits_name = lm_head_node.input[0]
        graph.node.remove(lm_head_node)
        
        # Add an Identity node to map the output of the decoder to 'pre_logits'
        identity_node = helper.make_node(
            'Identity',
            inputs=[pre_logits_name],
            outputs=['pre_logits']
        )
        graph.node.append(identity_node)
        
        # Replace output 'logits' with 'pre_logits'
        for out in graph.output:
            if out.name == 'logits':
                out.name = 'pre_logits'
                out.type.tensor_type.elem_type = TensorProto.FLOAT
                out.type.tensor_type.shape.dim[2].dim_value = 1024
                break
                
    onnx.save(model, output_path, save_as_external_data=True, all_tensors_to_one_file=True)
    print(f"Pruned decoder saved successfully.")

def build_cache_initializer(safetensors_path, output_path):
    print(f"Extracting cache projection weights from safetensors to create cache_initializer: {output_path}")
    from safetensors import safe_open
    from onnx import numpy_helper
    
    inputs = [helper.make_tensor_value_info('encoder_hidden_states', TensorProto.FLOAT, ['batch', 'encoder_seq', 1024])]
    outputs = []
    nodes = []
    initializers = []
    
    with safe_open(safetensors_path, framework="pt") as f:
        # 24 layers of encoder-decoder attention in NLLB 1.3B
        for i in range(24):
            # Outputs for key and value state arrays
            outputs.append(helper.make_tensor_value_info(f'present.{i}.encoder.key', TensorProto.FLOAT, ['batch', 16, 'encoder_seq', 64]))
            outputs.append(helper.make_tensor_value_info(f'present.{i}.encoder.value', TensorProto.FLOAT, ['batch', 16, 'encoder_seq', 64]))
            
            # Load weights and biases from safetensors
            k_weight = f.get_tensor(f"model.decoder.layers.{i}.encoder_attn.k_proj.weight").numpy()
            k_bias = f.get_tensor(f"model.decoder.layers.{i}.encoder_attn.k_proj.bias").numpy()
            v_weight = f.get_tensor(f"model.decoder.layers.{i}.encoder_attn.v_proj.weight").numpy()
            v_bias = f.get_tensor(f"model.decoder.layers.{i}.encoder_attn.v_proj.bias").numpy()
            
            # Transpose PyTorch linear weights for ONNX MatMul (out_features, in_features) -> (in_features, out_features)
            k_weight_transposed = k_weight.T
            v_weight_transposed = v_weight.T
            
            # Weight/bias initializer names
            k_weight_name = f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"
            k_bias_name = f"model.decoder.layers.{i}.encoder_attn.k_proj.bias"
            v_weight_name = f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"
            v_bias_name = f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"
            
            # Convert to ONNX initializers
            initializers.append(numpy_helper.from_array(k_weight_transposed, name=k_weight_name))
            initializers.append(numpy_helper.from_array(k_bias, name=k_bias_name))
            initializers.append(numpy_helper.from_array(v_weight_transposed, name=v_weight_name))
            initializers.append(numpy_helper.from_array(v_bias, name=v_bias_name))
            
            # Reshape shape constant
            reshape_shape_name = f'reshape_shape_{i}'
            reshape_shape_init = helper.make_tensor(reshape_shape_name, TensorProto.INT64, [4], [0, -1, 16, 64])
            initializers.append(reshape_shape_init)
            
            # k projection nodes
            nodes.append(helper.make_node(
                'MatMul', 
                inputs=['encoder_hidden_states', k_weight_name], 
                outputs=[f'k_proj_{i}']
            ))
            nodes.append(helper.make_node(
                'Add', 
                inputs=[f'k_proj_{i}', k_bias_name], 
                outputs=[f'k_proj_add_{i}']
            ))
            nodes.append(helper.make_node(
                'Reshape', 
                inputs=[f'k_proj_add_{i}', reshape_shape_name], 
                outputs=[f'k_proj_reshaped_{i}']
            ))
            nodes.append(helper.make_node(
                'Transpose', 
                inputs=[f'k_proj_reshaped_{i}'], 
                outputs=[f'present.{i}.encoder.key'], 
                perm=[0, 2, 1, 3]
            ))
            
            # v projection nodes
            nodes.append(helper.make_node(
                'MatMul', 
                inputs=['encoder_hidden_states', v_weight_name], 
                outputs=[f'v_proj_{i}']
            ))
            nodes.append(helper.make_node(
                'Add', 
                inputs=[f'v_proj_{i}', v_bias_name], 
                outputs=[f'v_proj_add_{i}']
            ))
            nodes.append(helper.make_node(
                'Reshape', 
                inputs=[f'v_proj_add_{i}', reshape_shape_name], 
                outputs=[f'v_proj_reshaped_{i}']
            ))
            nodes.append(helper.make_node(
                'Transpose', 
                inputs=[f'v_proj_reshaped_{i}'], 
                outputs=[f'present.{i}.encoder.value'], 
                perm=[0, 2, 1, 3]
            ))
            
    # Assemble Graph
    graph = helper.make_graph(
        nodes=nodes,
        name='cache_initializer',
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    
    model = helper.make_model(graph, producer_name='dhruva-converter', ir_version=10, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.save(model, output_path)
    print(f"Created cache_initializer model at: {output_path}")

def quantize_and_finalize():
    print("=== Step 3: Quantizing all components to INT8 ===")
    os.makedirs("final_int8_models", exist_ok=True)
    components = [
        "NLLB_1.3b_encoder.onnx",
        "NLLB_1.3b_decoder.onnx",
        "NLLB_1.3b_embed_and_lm_head.onnx",
        "NLLB_1.3b_cache_initializer.onnx"
    ]
    for comp in components:
        print(f"Quantizing {comp}...")
        quantize_dynamic(
            model_input=f"optimized_splits/{comp}",
            model_output=f"final_int8_models/{comp}",
            weight_type=QuantType.QUInt8
        )
    print("Quantization completed. Files saved to: final_int8_models/")

def run_interleaved_pipeline():
    print("=== STARTING MEMORY-OPTIMIZED INTERLEAVED PIPELINE ===")
    os.makedirs("optimized_splits", exist_ok=True)
    os.makedirs("final_int8_models", exist_ok=True)

    # Clean previous base model files
    if os.path.exists("nllb_base_onnx"):
        import shutil
        shutil.rmtree("nllb_base_onnx")

    # 1. Export the encoder
    print("\n--- Phase 1: Exporting Encoder ---")
    subprocess.run([sys.executable, sys.argv[0], "--export-encoder"], check=True)

    # 2. Process and Quantize Encoder / Embedding
    print("\n--- Phase 2: Processing and Quantizing Encoder and Embeddings ---")
    encoder_path = "nllb_base_onnx/encoder_model.onnx"
    encoder_model = onnx.load(encoder_path)
    embed_initializer = find_embedding_weight(encoder_model)
    if embed_initializer is None:
        print("Error: Could not locate the shared embedding weight matrix in encoder model.")
        sys.exit(1)

    print("Building NLLB_1.3b_embed_and_lm_head.onnx...")
    build_embed_and_lm_head(embed_initializer, "optimized_splits/NLLB_1.3b_embed_and_lm_head.onnx")
    print("Quantizing NLLB_1.3b_embed_and_lm_head.onnx...")
    quantize_dynamic(
        model_input="optimized_splits/NLLB_1.3b_embed_and_lm_head.onnx",
        model_output="final_int8_models/NLLB_1.3b_embed_and_lm_head.onnx",
        weight_type=QuantType.QUInt8
    )
    os.remove("optimized_splits/NLLB_1.3b_embed_and_lm_head.onnx")

    print("Pruning embedding from encoder...")
    prune_encoder_embedding(encoder_path, "optimized_splits/NLLB_1.3b_encoder.onnx", embed_initializer.name)
    print("Quantizing NLLB_1.3b_encoder.onnx...")
    quantize_dynamic(
        model_input="optimized_splits/NLLB_1.3b_encoder.onnx",
        model_output="final_int8_models/NLLB_1.3b_encoder.onnx",
        weight_type=QuantType.QUInt8
    )
    # Clean up intermediate encoder files
    print("Cleaning up intermediate encoder files...")
    os.remove("optimized_splits/NLLB_1.3b_encoder.onnx")
    if os.path.exists("optimized_splits/NLLB_1.3b_encoder.onnx_data"):
        os.remove("optimized_splits/NLLB_1.3b_encoder.onnx_data")
    import shutil
    shutil.rmtree("nllb_base_onnx") # Reclaim 2.9 GB immediately!

    # 3. Export the decoder
    print("\n--- Phase 3: Exporting Decoder ---")
    subprocess.run([sys.executable, sys.argv[0], "--export-decoder"], check=True)

    # 4. Process and Quantize Decoder
    print("\n--- Phase 4: Processing and Quantizing Decoder ---")
    decoder_path = "nllb_base_onnx/decoder_model.onnx"
    print("Pruning embedding and projection from decoder...")
    prune_decoder_embedding_and_lm_head(decoder_path, "optimized_splits/NLLB_1.3b_decoder.onnx", embed_initializer.name)
    print("Quantizing NLLB_1.3b_decoder.onnx...")
    quantize_dynamic(
        model_input="optimized_splits/NLLB_1.3b_decoder.onnx",
        model_output="final_int8_models/NLLB_1.3b_decoder.onnx",
        weight_type=QuantType.QUInt8
    )
    # Clean up intermediate decoder files
    print("Cleaning up intermediate decoder files...")
    os.remove("optimized_splits/NLLB_1.3b_decoder.onnx")
    if os.path.exists("optimized_splits/NLLB_1.3b_decoder.onnx_data"):
        os.remove("optimized_splits/NLLB_1.3b_decoder.onnx_data")
    shutil.rmtree("nllb_base_onnx") # Reclaim 5.4 GB immediately!

    # 5. Build Cache Initializer from Safetensors
    print("\n--- Phase 5: Building and Quantizing Cache Initializer ---")
    import glob
    hf_cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-1.3B/snapshots/*/model.safetensors")
    safetensors_paths = glob.glob(hf_cache_pattern)
    if not safetensors_paths:
        print("Error: Could not locate model.safetensors in huggingface cache.")
        sys.exit(1)
    safetensors_path = safetensors_paths[0]

    build_cache_initializer(safetensors_path, "optimized_splits/NLLB_1.3b_cache_initializer.onnx")
    print("Quantizing NLLB_1.3b_cache_initializer.onnx...")
    quantize_dynamic(
        model_input="optimized_splits/NLLB_1.3b_cache_initializer.onnx",
        model_output="final_int8_models/NLLB_1.3b_cache_initializer.onnx",
        weight_type=QuantType.QUInt8
    )
    os.remove("optimized_splits/NLLB_1.3b_cache_initializer.onnx")

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY! ===")
    print("Final quantized models are saved in: final_int8_models/")

def main():
    if len(sys.argv) < 2:
        run_interleaved_pipeline()
        sys.exit(0)
        
    mode = sys.argv[1]
    if mode == "--run-all":
        run_interleaved_pipeline()
    elif mode in ["--export", "--export-encoder", "--export-decoder"]:
        export_base_onnx()
    elif mode == "--split":
        # Load and optimize
        encoder_path = "nllb_base_onnx/encoder_model.onnx"
        decoder_path = "nllb_base_onnx/decoder_model.onnx"
        
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            print("Error: Base ONNX models not found in nllb_base_onnx/. Run with --export first.")
            sys.exit(1)
            
        import glob
        hf_cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-1.3B/snapshots/*/model.safetensors")
        safetensors_paths = glob.glob(hf_cache_pattern)
        if not safetensors_paths:
            print("Error: Could not locate model.safetensors in huggingface cache.")
            sys.exit(1)
        safetensors_path = safetensors_paths[0]
            
        os.makedirs("optimized_splits", exist_ok=True)
        encoder_model = onnx.load(encoder_path)
        embed_initializer = find_embedding_weight(encoder_model)
        
        if embed_initializer is None:
            print("Error: Could not locate the shared embedding weight matrix in encoder model.")
            sys.exit(1)
            
        build_embed_and_lm_head(embed_initializer, "optimized_splits/NLLB_1.3b_embed_and_lm_head.onnx")
        prune_encoder_embedding(encoder_path, "optimized_splits/NLLB_1.3b_encoder.onnx", embed_initializer.name)
        prune_decoder_embedding_and_lm_head(decoder_path, "optimized_splits/NLLB_1.3b_decoder.onnx", embed_initializer.name)
        build_cache_initializer(safetensors_path, "optimized_splits/NLLB_1.3b_cache_initializer.onnx")
        
    elif mode == "--quantize":
        quantize_and_finalize()

if __name__ == "__main__":
    main()
