#include <pybind11/pybind11.h>

#include "attention/aggregate_values/aggregate_values_crpe.h"
#include "attention/attention_logits/attention_logits_crpe.h"
#include "attention/query_key_pairs/stratified_qk_pairs.h"
#include "farthest_point_sampling/fps.h"
#include "neighbors/ball_query/ball_query.h"
#include "neighbors/knn_query/knn_query.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "farthestPointSampling",
        &farthestPointSampling,
        "Subsample vertices of point cloud using Farthest Point Sampling.",
        py::arg("coordinates"),
        py::arg("batch_counts"),
        py::arg("ratio"),
        py::arg("max_points")
    );
    m.def(
        "ballQuery",
        &ballQuery,
        "For each vertex of other point cloud, sample k neighbors of self point cloud within radius.",
        py::arg("self_coordinates"),
        py::arg("self_batch_counts"),
        py::arg("other_coordinates"),
        py::arg("other_batch_indices"),
        py::arg("k"),
        py::arg("radius")
    );
    m.def(
        "kNNQuery",
        &kNNQuery,
        "For each vertex of other point cloud, sample k nearest neighbors of self point cloud.",
        py::arg("self_coordinates"),
        py::arg("self_batch_counts"),
        py::arg("other_coordinates"),
        py::arg("other_batch_indices"),
        py::arg("k"),
        py::arg("sorted") = true
    );
    m.def(
        "attentionLogitsCRPE_forward",
        &attentionLogitsCRPE_forward,
        "Compute attention logits using contextual relative position encoding.",
        py::arg("queries"),
        py::arg("keys"),
        py::arg("query_key_pair_idxs"),
        py::arg("query_rel_xyz_tables"),
        py::arg("key_rel_xyz_tables"),
        py::arg("rel_xyz_table_idxs")
    );
    m.def(
        "attentionLogitsCRPE_backward",
        &attentionLogitsCRPE_backward,
        "Compute gradient of attention logits wrt inputs using contextual relative position encoding.",
        py::arg("queries"),
        py::arg("keys"),
        py::arg("query_key_pair_idxs"),
        py::arg("query_rel_xyz_tables"),
        py::arg("key_rel_xyz_tables"),
        py::arg("rel_xyz_table_idxs"),
        py::arg("attention_grad")
    );    
    m.def(
        "aggregateValuesCRPE_forward",
        &aggregateValuesCRPE_forward,
        "Aggregate values given attention distributions using contextual relative position encoding.",
        py::arg("values"),
        py::arg("query_key_pair_idxs"),
        py::arg("query_key_offsets"),
        py::arg("max_key_count"),
        py::arg("attention"),
        py::arg("value_rel_xyz_tables"),
        py::arg("rel_xyz_table_idxs")
    );
    m.def(
        "aggregateValuesCRPE_backward",
        &aggregateValuesCRPE_backward,
        "Compute gradient of aggregated values wrt inputs using contextual relative position encoding.",
        py::arg("values"),
        py::arg("query_key_pair_idxs"),
        py::arg("query_key_offsets"),
        py::arg("max_key_count"),
        py::arg("attention"),
        py::arg("value_rel_xyz_tables"),
        py::arg("rel_xyz_table_idxs"),
        py::arg("agg_values_grad")
    );
    m.def(
        "stratifiedQueryKeyPairs",
        &stratifiedQueryKeyPairs,
        "Compute indices of query-key pairs in small and stratified large windows.",
        py::arg("small_window_idxs"),
        py::arg("large_window_idxs"),
        py::arg("downsample_idxs"),
        py::arg("union") = true
    );
}
