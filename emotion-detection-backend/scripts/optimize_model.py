"""
Convert IndoBERT model to ONNX format and run benchmarks.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def convert_model():
    """Convert IndoBERT to ONNX."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    import shutil
    
    model_dir = Path(r"d:\project-emotion-detected-system\saved_models_indobert\indobert_emotion")
    output_dir = Path(r"d:\project-emotion-detected-system\saved_models_indobert\indobert_onnx")
    
    print("="*60)
    print("🔄 CONVERTING INDOBERT TO ONNX")
    print("="*60)
    print(f"Source: {model_dir}")
    print(f"Output: {output_dir}")
    
    # Convert
    print("\n📦 Loading and converting model...")
    start = time.time()
    
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir / "model"),
        export=True
    )
    
    # Save ONNX model
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    
    # Copy tokenizer
    print("📝 Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    tokenizer.save_pretrained(str(output_dir))
    
    # Copy label encoder
    print("🏷️ Copying label encoder...")
    shutil.copy(str(model_dir / "label_encoder.pkl"), str(output_dir / "label_encoder.pkl"))
    
    elapsed = time.time() - start
    print(f"\n✅ Conversion complete in {elapsed:.1f}s")
    
    # Check file sizes
    print("\n📊 File sizes:")
    original_size = sum(f.stat().st_size for f in (model_dir / "model").rglob("*") if f.is_file())
    onnx_size = sum(f.stat().st_size for f in output_dir.rglob("*.onnx") if f.is_file())
    
    print(f"   Original: {original_size / 1024 / 1024:.1f} MB")
    print(f"   ONNX: {onnx_size / 1024 / 1024:.1f} MB")
    print(f"   Reduction: {(1 - onnx_size/original_size)*100:.1f}%")
    
    return output_dir


def benchmark():
    """Run benchmark comparison."""
    import pickle
    import numpy as np
    from transformers import AutoTokenizer
    import onnxruntime as ort
    
    from app.ml.model_indobert import get_indobert_predictor
    
    model_dir = Path(r"d:\project-emotion-detected-system\saved_models_indobert\indobert_onnx")
    
    print("\n" + "="*60)
    print("🏁 BENCHMARK: PyTorch vs ONNX")
    print("="*60)
    
    # Test texts
    test_texts = [
        "Saya sangat senang hari ini!",
        "Sedih banget rasanya",
        "Marah sama kamu!",
        "Takut dengan keadaan ini",
        "Aku cinta kamu selamanya",
        "Hari ini biasa saja",
        "Alhamdulillah bahagia sekali",
        "Kecewa dengan hasilnya",
    ]
    
    iterations = 5
    
    # Load PyTorch model
    print("\n📦 Loading PyTorch model...")
    pytorch_predictor = get_indobert_predictor()
    if not pytorch_predictor.is_loaded():
        pytorch_predictor.load_model()
    
    # Load ONNX model
    print("📦 Loading ONNX model...")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    onnx_session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options,
        providers=['CPUExecutionProvider']
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    with open(str(model_dir / "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    def onnx_predict(text):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        inputs = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64)
        }
        outputs = onnx_session.run(None, inputs)
        return outputs[0]
    
    # Warm up
    print("\n🔥 Warming up...")
    for text in test_texts[:2]:
        pytorch_predictor.predict(text)
        onnx_predict(text)
    
    # Benchmark PyTorch
    print("\n📊 Benchmarking PyTorch...")
    pytorch_times = []
    for text in test_texts:
        for _ in range(iterations):
            start = time.perf_counter()
            pytorch_predictor.predict(text)
            elapsed = (time.perf_counter() - start) * 1000
            pytorch_times.append(elapsed)
    
    # Benchmark ONNX
    print("📊 Benchmarking ONNX...")
    onnx_times = []
    for text in test_texts:
        for _ in range(iterations):
            start = time.perf_counter()
            onnx_predict(text)
            elapsed = (time.perf_counter() - start) * 1000
            onnx_times.append(elapsed)
    
    # Results
    pytorch_avg = sum(pytorch_times) / len(pytorch_times)
    onnx_avg = sum(onnx_times) / len(onnx_times)
    
    print("\n" + "="*60)
    print("📈 RESULTS")
    print("="*60)
    print(f"{'Model':<15} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*60)
    print(f"{'PyTorch':<15} {pytorch_avg:<12.2f} {min(pytorch_times):<12.2f} {max(pytorch_times):<12.2f}")
    print(f"{'ONNX':<15} {onnx_avg:<12.2f} {min(onnx_times):<12.2f} {max(onnx_times):<12.2f}")
    print("-"*60)
    speedup = pytorch_avg / onnx_avg
    print(f"⚡ ONNX Speedup: {speedup:.2f}x faster!")
    print("="*60)
    
    return {
        "pytorch_avg_ms": pytorch_avg,
        "onnx_avg_ms": onnx_avg,
        "speedup": speedup
    }


if __name__ == "__main__":
    # Step 1: Convert to ONNX
    output_dir = convert_model()
    
    # Step 2: Run benchmark
    results = benchmark()
    
    print("\n✅ Optimization complete!")
    print(f"   ONNX model saved to: {output_dir}")
