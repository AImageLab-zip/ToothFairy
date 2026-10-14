from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='pointops',
    version='0.0.0',
    ext_modules=[CUDAExtension(
        name='pointops',
        sources=[
            'src/api.cpp',
            'src/attention/aggregate_values/aggregate_values_crpe.cu',
            'src/attention/aggregate_values/aggregate_values_crpe_cuda.cu',
            'src/attention/attention_logits/attention_logits_crpe.cu',
            'src/attention/attention_logits/attention_logits_crpe_cuda.cu',
            'src/attention/query_key_pairs/stratified_qk_pairs.cu',
            'src/attention/query_key_pairs/stratified_qk_pairs_cuda.cu',
            'src/farthest_point_sampling/fps.cu',
            'src/farthest_point_sampling/fps_cuda.cu',
            'src/neighbors/ball_query/ball_query.cu',
            'src/neighbors/ball_query/ball_query_cuda.cu',
            'src/neighbors/knn_query/knn_query.cu',
            'src/neighbors/knn_query/knn_query_cuda.cu',
        ],
        runtime_library_dirs=['/usr/local/lib'],
    )],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
