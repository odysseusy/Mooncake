#include "device_comm/device_collective/algorithms/ring/ring_kernel.cuh"

#include <cstdint>

#include "device_comm/device_collective/device_collective_kernel.cuh"
#include "device_comm/device_collective/protocols/simple/simple_primitives.cuh"
#include "device_comm/device_utils/device_utils.cuh"

namespace mooncake {
namespace {

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveResult
runRingChannel(const ReduceRequest& request, RingDeviceState* state,
               uint32_t channel, cooperative_groups::thread_block block) {
    const auto& view = *state->collective.group_view;
    // A captured request may outlive its root's membership.
    if (view.activeIndex(request.root) < 0) {
        return {
            .failed_rank = request.root,
            .failure_type = CollectiveFailureType::UnavailableParticipant,
        };
    }
    if (request.count == 0) return {};

    const auto& plan = state->plan;
    const uint32_t channel_count = RingChannelPartition::chooseChannelCount(
        request.count * sizeof(T), plan.workspace.bytes);
    if (channel >= channel_count) return {};
    const auto partition =
        RingChannelPartition::make(request.count, channel, channel_count);
    const bool is_root = view.self().in_group_rank == request.root;
    const bool is_first =
        view.participants[plan.predecessor_index].in_group_rank == request.root;
    const auto* input =
        static_cast<const T*>(request.send_buffer) + partition.offset;
    // Only the root has an output operand. Other ranks may pass nullptr.
    auto* output = is_root
                       ? static_cast<T*>(request.recv_buffer) + partition.offset
                       : nullptr;

    PG_ASSERT(state->simple);
    auto primitives = SimplePrimitives<T>::bind(
        *state->simple, plan.workspace, view, plan.predecessor_index,
        plan.successor_index, channel, channel_count, kMaxRingChunkBytes,
        state->collective.timeout_ticks);
    // Every rank exchanges workspace readiness. Reduction starts at the
    // root's successor and finishes at the root.
    PG_COLL_TRY(primitives.begin(block));
    for (uint64_t offset = 0; offset < partition.count;) {
        const CollectiveChunk<T> chunk{
            .source = input + offset,
            .destination = is_root ? output + offset : nullptr,
            .count =
                minimum(primitives.chunkElements(), partition.count - offset),
        };
        if (is_root) {
            PG_COLL_TRY(primitives.template recvReduceCopy<Op>(chunk, block));
        } else if (is_first) {
            PG_COLL_TRY(primitives.send(chunk, block));
        } else {
            PG_COLL_TRY(primitives.template recvReduceSend<Op>(chunk, block));
        }
        offset += chunk.count;
    }
    return primitives.finish(block);
}

template <typename T, ReduceOp Op>
__global__ __launch_bounds__(kRingBlockThreads,
                             1) void ringReduceKernel(ReduceRequest request,
                                                      RingDeviceState* state) {
    PG_ASSERT(state);
    runCollectiveInvocation<CollectiveExecution::MultiCTA>(
        request, &state->plan, state->collective,
        [&](cooperative_groups::thread_block block) {
            return runRingChannel<T, Op>(request, state, blockIdx.x, block);
        });
}

template <typename T, ReduceOp Op>
cudaError_t launchKernel(const ReduceRequest& request, RingDeviceState* state,
                         cudaStream_t stream) {
    const dim3 grid(kMaxDeviceCollectiveChannels);
    ringReduceKernel<T, Op>
        <<<grid, kRingBlockThreads, 0, stream>>>(request, state);
    return cudaGetLastError();
}

template <typename T>
cudaError_t launchForType(const ReduceRequest& request, RingDeviceState* state,
                          cudaStream_t stream) {
    switch (request.op) {
        case ReduceOp::Sum:
            return launchKernel<T, ReduceOp::Sum>(request, state, stream);
        case ReduceOp::Product:
            return launchKernel<T, ReduceOp::Product>(request, state, stream);
        case ReduceOp::Min:
            return launchKernel<T, ReduceOp::Min>(request, state, stream);
        case ReduceOp::Max:
            return launchKernel<T, ReduceOp::Max>(request, state, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t launchRingReduceKernel(const ReduceRequest& request,
                                   RingDeviceState* state,
                                   cudaStream_t stream) {
    switch (request.datatype) {
        case DataType::Float16:
            return launchForType<__half>(request, state, stream);
        case DataType::Uint8:
            return launchForType<uint8_t>(request, state, stream);
        case DataType::Int8:
            return launchForType<int8_t>(request, state, stream);
        case DataType::Int16:
            return launchForType<int16_t>(request, state, stream);
        case DataType::Int32:
            return launchForType<int32_t>(request, state, stream);
        case DataType::Int64:
            return launchForType<int64_t>(request, state, stream);
        case DataType::Bfloat16:
            return launchForType<__nv_bfloat16>(request, state, stream);
        case DataType::Float32:
            return launchForType<float>(request, state, stream);
        case DataType::Float64:
            return launchForType<double>(request, state, stream);
        case DataType::Bool:
            return launchForType<bool>(request, state, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace mooncake
