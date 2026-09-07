# 1. 预处理（多个目录可重复 --input_dir，递归）
# 限制 BLAS/OMP 线程数：16 个 worker 进程并行时避免线程超额订阅拖慢整体
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python step_preprocess.py --input_dir ./data/mechcad --output_dir ./data/mechcad_processed --workers 16