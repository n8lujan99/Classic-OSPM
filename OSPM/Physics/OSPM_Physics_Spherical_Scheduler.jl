# ========================================================================================================================
# OSPM_Physics_Spherical_Scheduler.jl
#
# Shared worker-pool primitives for spherical-model scheduling.
# Model admission and batch dispatch remain in OSPM_Physics_Spherical.jl
# until the scheduler state is separated from evaluate_batch_theta.
# ========================================================================================================================

mutable struct SharedWorkerPool
    total::Int
    leased::Threads.Atomic{Int}
    priority_waiters::Threads.Atomic{Int}
end

function SharedWorkerPool(total::Int)
    total > 0 || error("SharedWorkerPool total must be positive")
    return SharedWorkerPool(total, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
end

@inline function _try_acquire_worker!(pool::SharedWorkerPool; priority::Bool=false)
    while true
        leased = pool.leased[]
        reserved = priority ? 0 : min(pool.priority_waiters[], pool.total)
        leased >= pool.total - reserved && return false
        Threads.atomic_cas!(pool.leased, leased, leased + 1) == leased && return true
    end
end

function _acquire_worker!(pool::SharedWorkerPool; priority::Bool=false)
    priority && Threads.atomic_add!(pool.priority_waiters, 1)
    try
        while !_try_acquire_worker!(pool; priority=priority)
            yield()
        end
    finally
        priority && Threads.atomic_add!(pool.priority_waiters, -1)
    end
    return nothing
end

@inline function _release_worker!(pool::SharedWorkerPool)
    previous = Threads.atomic_add!(pool.leased, -1)
    previous > 0 || error("Shared worker pool lease underflow")
    return nothing
end
