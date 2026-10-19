#include "device_comm/device_collective/algorithms/one_shot/one_shot_kernel.cuh"

#include <cstdint>

#include "device_comm/device_collective/device_collective_kernel.cuh"

namespace mooncake {
namespace {

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveResult runOneShotChannel(
    const ReduceScatterRequest& request, OneShotDeviceState* state,
    uint32_t channel, uint32_t channel_count,
    cooperative_groups::thread_block block) {
    const auto& plan = state->plan;
    const auto& view = *state->collective.group_view;
    const auto partition =
        OneShotChannelPartition::make<T>(request.count, channel, channel_count);
    if (partition.count == 0) return {};

    const auto* input =
        static_cast<const T*>(request.send_buffer) + partition.offset;
    const auto* local_input =
        input +
        static_cast<uint64_t>(view.self().in_group_rank) * request.count;
    auto* output = static_cast<T*>(request.recv_buffer) + partition.offset;
    PG_ASSERT(state->ll);
    OneShotExchange<T> exchange(view, plan, *state->ll, channel,
                                state->collective.timeout_ticks);
    uint64_t offset = 0;
    while (offset < partition.count) {
        PG_COLL_TRY(exchange.begin(block));
        const CollectiveChunk<T> chunk{
            .source = local_input + offset,
            .destination = output + offset,
            .count = minimum(partition.count - offset,
                             exchange.chunkElementCapacity()),
        };
        // Send each peer its in-group-rank input segment before writing local
        // output.
        exchange.sendAll(input + offset, chunk.count, request.count, block);
        const auto failed = exchange.template reducePeers<Op>(chunk, block);
        PG_COLL_TRY(exchange.finish(failed, block));
        offset += chunk.count;
    }
    return {};
}

template <typename T, ReduceOp Op, CollectiveExecution Execution>
__global__
__launch_bounds__(kOneShotMaxThreads) void oneShotReduceScatterKernel(
    ReduceScatterRequest request, OneShotDeviceState* state) {
    PG_ASSERT(state);
    runCollectiveInvocation<Execution>(
        request, &state->plan, state->collective,
        [&](cooperative_groups::thread_block block) {
            return runOneShotChannel<T, Op>(request, state, blockIdx.x,
                                            gridDim.x, block);
        });
}

template <typename T, ReduceOp Op>
cudaError_t launchKernel(const ReduceScatterRequest& request,
                         OneShotDeviceState* state, int threads,
                         uint32_t channels, cudaStream_t stream) {
    if (channels > 1) {
        oneShotReduceScatterKernel<T, Op, CollectiveExecution::MultiCTA>
            <<<channels, threads, 0, stream>>>(request, state);
    } else {
        oneShotReduceScatterKernel<T, Op, CollectiveExecution::SingleCTA>
            <<<1, threads, 0, stream>>>(request, state);
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t launchForType(const ReduceScatterRequest& request,
                          OneShotDeviceState* state, int threads,
                          uint32_t channels, cudaStream_t stream) {
    switch (request.op) {
        case ReduceOp::Sum:
            return launchKernel<T, ReduceOp::Sum>(request, state, threads,
                                                  channels, stream);
        case ReduceOp::Product:
            return launchKernel<T, ReduceOp::Product>(request, state, threads,
                                                      channels, stream);
        case ReduceOp::Min:
            return launchKernel<T, ReduceOp::Min>(request, state, threads,
                                                  channels, stream);
        case ReduceOp::Max:
            return launchKernel<T, ReduceOp::Max>(request, state, threads,
                                                  channels, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t launchOneShotReduceScatterKernel(
    const ReduceScatterRequest& request, OneShotDeviceState* state,
    cudaStream_t stream) {
    if (!isDeviceReductionCombinationSupported(request.datatype, request.op))
        return cudaErrorInvalidValue;
    // The per-rank result size is fixed in a captured call.
    const uint64_t bytes = request.count * elementSize(request.datatype);
    const int threads = bytes <= 64    ? 128
                        : bytes <= 128 ? 256
                                       : kOneShotMaxThreads;
    const uint32_t channels = bytes <= 2048 ? 1 : kOneShotMaxChannels;
    switch (request.datatype) {
        case DataType::Float16:
            return launchForType<__half>(request, state, threads, channels,
                                         stream);
        case DataType::Uint8:
            return launchForType<uint8_t>(request, state, threads, channels,
                                          stream);
        case DataType::Int8:
            return launchForType<int8_t>(request, state, threads, channels,
                                         stream);
        case DataType::Int16:
            return launchForType<int16_t>(request, state, threads, channels,
                                          stream);
        case DataType::Int64:
            return launchForType<int64_t>(request, state, threads, channels,
                                          stream);
        case DataType::Bfloat16:
            return launchForType<__nv_bfloat16>(request, state, threads,
                                                channels, stream);
        case DataType::Float32:
            return launchForType<float>(request, state, threads, channels,
                                        stream);
        case DataType::Float64:
            return launchForType<double>(request, state, threads, channels,
                                         stream);
        case DataType::Int32:
            return launchForType<int32_t>(request, state, threads, channels,
                                          stream);
        case DataType::Bool:
            return launchForType<bool>(request, state, threads, channels,
                                       stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace mooncake
